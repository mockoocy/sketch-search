import gc
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sktr.config.config import CFG, DEVICE
from sktr.config.config_model import EarlyStopConfig
from sktr.logger import train_logger
from sktr.metrics import compute_map_at_k_chunked
from sktr.model import (
    Embedder,
    SupConLoss,
    TimmBackbone,
    build_photo_transform_eval,
    build_photo_transform_train,
    build_sketch_transform_eval,
    build_sketch_transform_train,
    dcl_loss,
)
from sktr.photo_sketch_dataset import (
    build_loader,
    get_paired_samples,
    get_samples_from_directories,
)
from sktr.type_defs import Sample


def cuda_cleanup() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def pairviews(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0] // 2
    return x.reshape(2, b, *x.shape[1:]).transpose(0, 1).contiguous()


@torch.no_grad()
def create_class_mask(labels: list[str]) -> torch.Tensor:
    labels_arr = np.array(labels)
    return torch.from_numpy(labels_arr[:, None] == labels_arr[None, :])


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    mean_average_precision_at_10: float
    precision_at_10: float
    mean_average_precision_at_30: float
    precision_at_30: float


@dataclass(frozen=True)
class ProjectorPayload:
    photo_embeddings_cpu: torch.Tensor
    sketch_embeddings_cpu: torch.Tensor
    photo_labels_cpu: torch.Tensor
    sketch_labels_cpu: torch.Tensor


@dataclass(frozen=True)
class EvalResult:
    metrics: EvaluationMetrics
    projector: ProjectorPayload | None


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor


@dataclass(frozen=True)
class StepState:
    phase_name: str
    epoch: int
    step_in_epoch: int
    global_step: int
    loss: float
    lr: float


class Callback(Protocol):
    def on_run_start(self, run_name: str) -> None: ...
    def on_phase_start(
        self,
        phase_name: str,
        start_step: int,
        total_steps: int,
    ) -> None: ...
    def on_step_end(self, state: StepState) -> None: ...
    def on_eval_end(
        self,
        phase_name: str,
        global_step: int,
        result: EvalResult,
        model: nn.Module,
    ) -> None: ...
    def on_epoch_end(self, phase_name: str, epoch: int, model: nn.Module) -> None: ...
    def on_phase_end(self, phase_name: str, end_step: int) -> None: ...
    def on_run_end(self) -> None: ...


class CallbackList:
    def __init__(self, callbacks: Iterable[Callback]) -> None:
        self._callbacks = list(callbacks)

    def on_run_start(self, run_name: str) -> None:
        for cb in self._callbacks:
            cb.on_run_start(run_name)

    def on_phase_start(
        self,
        phase_name: str,
        start_step: int,
        total_steps: int,
    ) -> None:
        for cb in self._callbacks:
            cb.on_phase_start(phase_name, start_step, total_steps)

    def on_step_end(self, state: StepState) -> None:
        for cb in self._callbacks:
            cb.on_step_end(state)

    def on_eval_end(
        self,
        phase_name: str,
        global_step: int,
        result: EvalResult,
        model: nn.Module,
    ) -> None:
        for cb in self._callbacks:
            cb.on_eval_end(phase_name, global_step, result, model)

    def on_epoch_end(self, phase_name: str, epoch: int, model: nn.Module) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(phase_name, epoch, model)

    def on_phase_end(self, phase_name: str, end_step: int) -> None:
        for cb in self._callbacks:
            cb.on_phase_end(phase_name, end_step)

    def on_run_end(self) -> None:
        for cb in self._callbacks:
            cb.on_run_end()


class StopTrainingError(Exception):
    pass


class EarlyStopCallback(Callback):
    def __init__(self, cfg: EarlyStopConfig) -> None:
        self.cfg = cfg
        self._best: float | None = None
        self._bad_evals = 0
        self._evals_seen = 0
        self.should_stop = False

    def _read_metric(self, result: EvalResult) -> float:
        m = result.metrics
        if self.cfg.monitor == "loss":
            return float(m.loss)
        if self.cfg.monitor == "mAP@10":
            return float(m.mean_average_precision_at_10)
        if self.cfg.monitor == "mAP@30":
            return float(m.mean_average_precision_at_30)
        if self.cfg.monitor == "P@10":
            return float(m.precision_at_10)
        if self.cfg.monitor == "P@30":
            return float(m.precision_at_30)
        err_msg = f"Unknown monitor metric: {self.cfg.monitor}"
        raise ValueError(err_msg)

    def _is_improvement(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.cfg.mode == "max":
            return value > (self._best + self.cfg.min_delta)
        if self.cfg.mode == "min":
            return value < (self._best - self.cfg.min_delta)
        err_msg = f"Unknown mode: {self.cfg.mode}"
        raise ValueError(err_msg)

    def on_eval_end(
        self,
        phase_name: str,
        global_step: int,
        result: EvalResult,
        model: nn.Module,  # noqa: ARG002
    ) -> None:
        self._evals_seen += 1
        if self._evals_seen <= self.cfg.warmup_evals:
            return

        value = self._read_metric(result)

        if self._is_improvement(value):
            self._best = value
            self._bad_evals = 0
            return

        self._bad_evals += 1
        if self._bad_evals >= self.cfg.patience:
            self.should_stop = True
            exception_messsage = (
                f"Early stopping on {phase_name} at step={global_step + 1}: "
                f"{self.cfg.monitor} did not improve for {self.cfg.patience} evals "
                f"(best={self._best}, last={value})."
            )
            raise StopTrainingError(exception_messsage)


def log_projector_embeddings(
    writer: SummaryWriter,
    payload: ProjectorPayload,
    tag: str,
    max_points: int = 2000,
    seed: int = 42,
) -> None:
    pe = payload.photo_embeddings_cpu
    se = payload.sketch_embeddings_cpu
    photo_labels = payload.photo_labels_cpu
    sketch_labels = payload.sketch_labels_cpu

    x = torch.cat([pe, se], dim=0)
    y = torch.cat([photo_labels, sketch_labels], dim=0)

    n_photo = pe.size(0)
    modality = torch.cat(
        [
            torch.zeros(n_photo, dtype=torch.long),
            torch.ones(se.size(0), dtype=torch.long),
        ],
        dim=0,
    )

    if x.size(0) > max_points:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(x.size(0), generator=g)[:max_points]
        x = x[idx]
        y = y[idx]
        modality = modality[idx]

    meta = [
        [str(int(lbl)), "photo" if int(mod) == 0 else "sketch"]
        for lbl, mod in zip(y.tolist(), modality.tolist(), strict=False)
    ]

    writer.add_embedding(
        x,
        metadata=meta,
        metadata_header=["category_id", "modality"],
        tag=tag,
        global_step=0,
    )


class TensorBoardCallback(Callback):
    def __init__(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def on_step_end(self, state: StepState) -> None:
        if (state.global_step + 1) % 50 != 0:
            return
        self.writer.add_scalar(
            f"{state.phase_name}/train_loss",
            state.loss,
            state.global_step,
        )
        self.writer.add_scalar(f"{state.phase_name}/lr", state.lr, state.global_step)

    def on_eval_end(
        self,
        phase_name: str,  # noqa: ARG002
        global_step: int,
        result: EvalResult,
        model: nn.Module,  # noqa: ARG002
    ) -> None:
        m = result.metrics
        self.writer.add_scalar("eval/loss", m.loss, global_step)
        self.writer.add_scalar(
            "eval/mAP@10",
            m.mean_average_precision_at_10,
            global_step,
        )
        self.writer.add_scalar("eval/Precision@10", m.precision_at_10, global_step)
        self.writer.add_scalar(
            "eval/mAP@30",
            m.mean_average_precision_at_30,
            global_step,
        )
        self.writer.add_scalar("eval/Precision@30", m.precision_at_30, global_step)
        if result.projector is not None:
            tag = f"eval/projector_step_{global_step + 1}"
            log_projector_embeddings(self.writer, result.projector, tag=tag)

    def on_epoch_end(self, phase_name: str, epoch: int, model: nn.Module) -> None:  # noqa: ARG002
        return

    def on_phase_end(self, phase_name: str, end_step: int) -> None:  # noqa: ARG002
        return

    def on_run_end(self) -> None:
        self.writer.close()


class ConsoleCallback(Callback):
    def on_phase_start(
        self,
        phase_name: str,  # noqa: ARG002
        start_step: int,  # noqa: ARG002
        total_steps: int,  # noqa: ARG002
    ) -> None:
        return

    def on_eval_end(
        self,
        phase_name: str,  # noqa: ARG002
        global_step: int,
        result: EvalResult,
        model: nn.Module,  # noqa: ARG002
    ) -> None:
        m = result.metrics
        tqdm.write(
            f"[Eval step {global_step + 1}] "
            f"loss={m.loss:.4f} "
            f"mAP@10={m.mean_average_precision_at_10:.4f} "
            f"P@10={m.precision_at_10:.4f} "
            f"mAP@30={m.mean_average_precision_at_30:.4f} "
            f"P@30={m.precision_at_30:.4f}",
        )


class CheckpointCallback(Callback):
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.best_map10 = float("-inf")

    def on_run_start(self, run_name: str) -> None:  # noqa: ARG002
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_eval_end(
        self,
        phase_name: str,
        global_step: int,  # noqa: ARG002
        result: EvalResult,
        model: nn.Module,
    ) -> None:
        if phase_name != "phase2":
            return
        score = result.metrics.mean_average_precision_at_10
        if score > self.best_map10:
            self.best_map10 = score
            torch.save(model.state_dict(), self.out_dir / "best_phase2_by_map10.pth")

    def on_epoch_end(self, phase_name: str, epoch: int, model: nn.Module) -> None:
        torch.save(
            model.state_dict(),
            self.out_dir / f"{phase_name}_epoch_{epoch + 1}.pth",
        )


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if CFG.training.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.05, fused=True)
    return torch.optim.Adam(params, lr=lr)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Evaluator:
    def __init__(self, loader: DataLoader[Sample]) -> None:
        self.loader = loader

    @torch.no_grad()
    def run(
        self,
        model: Embedder,
        loss_fn: SupConLoss,
        *,
        include_projector: bool,
    ) -> EvalResult:
        model.eval()

        total_loss = 0.0
        total_count = 0

        n = len(self.loader.dataset)
        photo_embeddings: torch.Tensor | None = None
        sketch_embeddings: torch.Tensor | None = None

        photo_category_ids = torch.empty(n, dtype=torch.long, device=DEVICE)
        sketch_category_ids = torch.empty(n, dtype=torch.long, device=DEVICE)

        label_to_id: dict[str, int] = {}
        idx = 0

        def to_ids(names: list[str]) -> torch.Tensor:
            ids: list[int] = []
            for s in names:
                if s not in label_to_id:
                    label_to_id[s] = len(label_to_id)
                ids.append(label_to_id[s])
            return torch.tensor(ids, device=DEVICE, dtype=torch.long)

        for batch in tqdm(self.loader, desc="Evaluating", leave=False):
            photos = batch["photo"].to(DEVICE, non_blocking=True)
            sketches = batch["sketch"].to(DEVICE, non_blocking=True)

            pe, se = model(photos, sketches)
            ids = to_ids(batch["categories"])
            mask = ids.unsqueeze(0) == ids.unsqueeze(1)
            loss = loss_fn(pairviews(torch.cat([pe, se], dim=0)), mask=mask)

            bsz = pe.size(0)
            total_loss += float(loss) * bsz
            total_count += bsz

            if photo_embeddings is None:
                d = pe.size(1)
                photo_embeddings = torch.empty(n, d, device=DEVICE, dtype=pe.dtype)
                sketch_embeddings = torch.empty(n, d, device=DEVICE, dtype=se.dtype)

            photo_embeddings[idx : idx + bsz].copy_(pe)
            sketch_embeddings[idx : idx + bsz].copy_(se)
            photo_category_ids[idx : idx + bsz].copy_(ids)
            sketch_category_ids[idx : idx + bsz].copy_(ids)
            idx += bsz

        val_loss = total_loss / total_count if total_count > 0 else float("inf")

        metrics_k = compute_map_at_k_chunked(
            query_emb=sketch_embeddings,
            query_labels=sketch_category_ids,
            gallery_emb=photo_embeddings,
            gallery_labels=photo_category_ids,
            ks=(10, 30),
            chunk_size=128,
        )

        map10, p10 = metrics_k[10]
        map30, p30 = metrics_k[30]

        projector: ProjectorPayload | None = None
        if include_projector:
            projector = ProjectorPayload(
                photo_embeddings_cpu=photo_embeddings.detach().cpu(),
                sketch_embeddings_cpu=sketch_embeddings.detach().cpu(),
                photo_labels_cpu=photo_category_ids.detach().cpu(),
                sketch_labels_cpu=sketch_category_ids.detach().cpu(),
            )

        del photo_embeddings, sketch_embeddings, photo_category_ids, sketch_category_ids
        cuda_cleanup()

        return EvalResult(
            metrics=EvaluationMetrics(
                loss=val_loss,
                mean_average_precision_at_10=map10,
                precision_at_10=p10,
                mean_average_precision_at_30=map30,
                precision_at_30=p30,
            ),
            projector=projector,
        )


class Phase(Protocol):
    name: str

    def epochs(self) -> int: ...
    def train_loader(self) -> DataLoader[Sample]: ...
    def warmup_steps(self) -> int: ...
    def lr(self) -> float: ...

    def min_lr_ratio(self) -> float: ...

    def train_step(self, model: Embedder, batch: dict[str, Any]) -> StepResult: ...

    def should_eval(
        self,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        steps_in_phase: int,
    ) -> bool: ...

    def eval_step(
        self,
        model: Embedder,
        global_step: int,
        steps_in_phase: int,
    ) -> EvalResult: ...


class Phase1Dcl(Phase):
    def __init__(self, loader: DataLoader[Sample], evaluator: Evaluator) -> None:
        self.name = "phase1"
        self._loader = loader
        self._temperature = float(CFG.training.phase_1.temperature)
        self._evaluator = evaluator

    def epochs(self) -> int:
        return int(CFG.training.phase_1.epochs)

    def train_loader(self) -> DataLoader[Sample]:
        return self._loader

    def make_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        return build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(CFG.training.phase_1.warmup_steps),
            min_lr_ratio=float(CFG.training.phase_1.min_lr_ratio),
        )

    def train_step(self, model: Embedder, batch: dict[str, Any]) -> StepResult:
        photo = batch["photo"].to(DEVICE, non_blocking=True)
        sketch = batch["sketch"].to(DEVICE, non_blocking=True)

        pe, se = model(photo, sketch)
        loss = dcl_loss(pe, se, temperature=self._temperature)

        return StepResult(loss=loss)

    def should_eval(
        self,
        epoch: int,  # noqa: ARG002
        step_in_epoch: int,
        global_step: int,  # noqa: ARG002
        steps_in_phase: int,
    ) -> bool:
        return step_in_epoch == (steps_in_phase - 1)

    def eval_step(
        self,
        model: Embedder,
        global_step: int,  # noqa: ARG002
        steps_in_phase: int,  # noqa: ARG002
    ) -> EvalResult:
        return self._evaluator.run(
            model,
            SupConLoss(temperature=self._temperature),
            include_projector=False,
        )

    def warmup_steps(self) -> int:
        return CFG.training.phase_1.warmup_steps

    def lr(self) -> float:
        return CFG.training.phase_1.base_lr

    def min_lr_ratio(self) -> float:
        return CFG.training.phase_1.min_lr_ratio


class Phase2SupCon(Phase):
    def __init__(self, train_loader: DataLoader[Sample], evaluator: Evaluator) -> None:
        self.name = "phase2"
        self._train_loader = train_loader
        self._loss_fn = SupConLoss(temperature=float(CFG.training.phase_2.temperature))
        self._evaluator = evaluator
        self._eval_every = int(CFG.validation.eval_every_steps)

    def epochs(self) -> int:
        return int(CFG.training.phase_2.epochs)

    def train_loader(self) -> DataLoader[Sample]:
        return self._train_loader

    def make_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        return build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=CFG.training.phase_2.warmup_steps,
            min_lr_ratio=CFG.training.phase_2.min_lr_ratio,
        )

    def train_step(self, model: Embedder, batch: dict[str, Any]) -> StepResult:
        photo = batch["photo"].to(DEVICE, non_blocking=True)
        sketch = batch["sketch"].to(DEVICE, non_blocking=True)

        mask = create_class_mask(batch["categories"]).to(DEVICE)

        pe, se = model(photo, sketch)
        loss = self._loss_fn(pairviews(torch.cat([pe, se], dim=0)), mask=mask)

        return StepResult(loss=loss)

    def should_eval(
        self,
        epoch: int,  # noqa: ARG002
        step_in_epoch: int,
        global_step: int,
        steps_in_phase: int,
    ) -> bool:
        is_first = global_step == 0
        is_eval_every = (global_step + 1) % self._eval_every == 0
        is_epoch_end = step_in_epoch == (steps_in_phase - 1)
        is_last = global_step == (self.epochs() * steps_in_phase - 1)
        return is_first or is_eval_every or is_epoch_end or is_last

    def eval_step(
        self,
        model: Embedder,
        global_step: int,
        steps_in_phase: int,
    ) -> EvalResult:
        is_last = global_step == (self.epochs() * steps_in_phase - 1)
        include_projector = is_last
        return self._evaluator.run(
            model,
            self._loss_fn,
            include_projector=include_projector,
        )

    def warmup_steps(self) -> int:
        return CFG.training.phase_2.warmup_steps

    def lr(self) -> float:
        return CFG.training.phase_2.base_lr

    def min_lr_ratio(self) -> float:
        return CFG.training.phase_2.min_lr_ratio


class Engine:
    def __init__(self, model: Embedder, callbacks: Iterable[Callback]) -> None:
        self.model = model
        self.cbs = CallbackList(callbacks)

    def run(self, run_name: str, phases: list[Phase], start_step: int = 0) -> int:
        self.cbs.on_run_start(run_name)

        global_step = start_step
        for phase in phases:
            epochs = phase.epochs()
            if epochs <= 0:
                continue

            loader = phase.train_loader()
            steps_in_epoch = len(loader)
            total_steps = epochs * steps_in_epoch

            optimizer = make_optimizer(self.model, lr=phase.lr())
            scheduler = phase.make_scheduler(optimizer, total_steps=total_steps)
            scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

            self.model.train()
            self.cbs.on_phase_start(phase.name, global_step, total_steps)

            for epoch in range(epochs):
                pbar = tqdm(
                    loader,
                    desc=f"[{phase.name}] Epoch {epoch + 1}/{epochs}",
                    miniters=50,
                )
                for step_in_epoch, batch in enumerate(pbar):
                    optimizer.zero_grad(set_to_none=True)

                    out = phase.train_step(self.model, batch)

                    scaler.scale(out.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    lr = float(optimizer.param_groups[0]["lr"])
                    loss_val = float(out.loss.detach())
                    state = StepState(
                        phase_name=phase.name,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        loss=loss_val,
                        lr=lr,
                    )
                    self.cbs.on_step_end(state)

                    if (global_step + 1) % 50 == 0:
                        pbar.set_postfix({"loss": loss_val})

                    if phase.should_eval(
                        epoch,
                        step_in_epoch,
                        global_step,
                        steps_in_epoch,
                    ):
                        result = phase.eval_step(
                            self.model,
                            global_step,
                            steps_in_epoch,
                        )
                        if result is not None:
                            self.cbs.on_eval_end(
                                phase.name,
                                global_step,
                                result,
                                self.model,
                            )
                        self.model.train()

                    global_step += 1

                self.cbs.on_epoch_end(phase.name, epoch, self.model)

            self.cbs.on_phase_end(phase.name, global_step)

            del optimizer, scheduler, scaler
            cuda_cleanup()

        self.cbs.on_run_end()
        return global_step


def build_model() -> Embedder:
    encoder = TimmBackbone(name=CFG.skitter.encoder_name).to(DEVICE)
    encoder.eval()

    model = Embedder(
        backbone=encoder,
        hidden_layer_size=CFG.skitter.projection_head_size,
        embedding_size=CFG.skitter.embedding_size,
    ).to(DEVICE)

    return torch.compile(model, mode="reduce-overhead")


def build_phase1_loader() -> DataLoader[Sample] | None:
    epochs = int(CFG.training.phase_1.epochs)
    if epochs <= 0:
        return None

    phase1_samples = get_paired_samples(
        images_root=Path(CFG.training.phase_1.images_path),
        sketches_root=Path(CFG.training.phase_1.sketches_path),
        fraction=float(CFG.training.phase_1.fraction_of_samples),
    )

    return build_loader(
        samples=phase1_samples,
        use_class_balanced_sampler=False,
        batch_size=int(CFG.training.batch_size),
        shuffle=True,
        photo_transform=build_photo_transform_train(),
        sketch_transform=build_sketch_transform_train(),
        num_workers=CFG.training.num_workers,
        persistent_workers=True,
        drop_last=True,
    )


def build_phase2_loaders() -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    train_samples, val_samples = get_samples_from_directories(
        images_root=Path(CFG.training.phase_2.images_path),
        sketches_root=Path(CFG.training.phase_2.sketches_path),
        per_category_fraction=CFG.training.phase_2.fraction_of_samples,
        val_fraction=CFG.validation.validation_fraction,
    )

    train_loader = build_loader(
        train_samples,
        batch_size=CFG.training.batch_size,
        photo_transform=build_photo_transform_train(),
        sketch_transform=build_sketch_transform_train(),
        num_workers=CFG.training.num_workers,
        samples_per_class=CFG.training.phase_2.samples_per_class,
    )

    val_loader: DataLoader[Sample] = build_loader(
        samples=val_samples,
        use_class_balanced_sampler=False,
        batch_size=CFG.training.batch_size,
        photo_transform=build_photo_transform_eval(),
        sketch_transform=build_sketch_transform_eval(),
        drop_last=False,
        num_workers=CFG.training.num_workers,
        prefetch_factor=2,
        persistent_workers=False,
        shuffle=False,
    )

    return train_loader, val_loader


def train() -> None:
    run_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(CFG.training.model_save_path) / run_name
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    callbacks: list[Callback] = [
        TensorBoardCallback(writer),
        ConsoleCallback(),
        CheckpointCallback(out_dir),
        EarlyStopCallback(CFG.training.early_stopping),
    ]

    model = build_model()

    phases: list[Phase] = []

    train_loader, val_loader = build_phase2_loaders()
    # evaluator will be used in both phases
    # but "phase 2" semantic makes sense, as it will
    # be used in a class-based manner
    evaluator = Evaluator(val_loader)
    if CFG.training.phase_1.enabled:
        phase1_loader = build_phase1_loader()
        phases.append(Phase1Dcl(phase1_loader, evaluator))
    if CFG.training.phase_2.enabled:
        phases.append(Phase2SupCon(train_loader, evaluator))

    engine = Engine(model, callbacks)
    try:
        engine.run(run_name, phases, start_step=0)
    except StopTrainingError as e:
        train_logger.info("Training stopped: %s", e)
