import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sktr.metrics import compute_map_at_k_chunked

from sktr.config.config import CFG, DEVICE
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
    get_qmul_paired_samples,
    get_samples_from_directories,
)
from sktr.type_defs import Sample

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


@dataclass
class EvaluationMetrics:
    loss: float
    mean_average_precision_at_10: float
    precision_at_10: float
    mean_average_precision_at_30: float
    precision_at_30: float


def log_projector_embeddings(
    writer: SummaryWriter,
    pe: torch.Tensor,
    se: torch.Tensor,
    photo_labels: torch.Tensor,
    sketch_labels: torch.Tensor,
    *,
    tag: str = "eval/projector",
    max_points: int = 2000,
    seed: int = 42,
) -> None:
    x = torch.cat([pe, se], dim=0)
    y = torch.cat([photo_labels, sketch_labels], dim=0)

    n_photo = pe.size(0)
    modality = torch.cat(
        [
            torch.zeros(n_photo, dtype=torch.long, device=x.device),
            torch.ones(se.size(0), dtype=torch.long, device=x.device),
        ],
        dim=0,
    )

    if x.size(0) > max_points:
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        idx = torch.randperm(x.size(0), generator=g, device=x.device)[:max_points]
        x = x[idx]
        y = y[idx]
        modality = modality[idx]

    meta = [
        [str(int(lbl)), "photo" if int(mod) == 0 else "sketch"]
        for lbl, mod in zip(y.cpu().tolist(), modality.cpu().tolist(), strict=False)
    ]

    writer.add_embedding(
        x.cpu(),
        metadata=meta,
        metadata_header=["category_id", "modality"],
        tag=tag,
        global_step=0,  # <- critical
    )


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


def pairviews(x: torch.Tensor) -> torch.Tensor:
    # [2B,D] -> [B,2,D]
    b = x.shape[0] // 2
    return x.reshape(2, b, *x.shape[1:]).transpose(0, 1).contiguous()


@torch.no_grad()
def _create_class_mask(labels: list[str]) -> torch.Tensor:
    labels_arr = np.array(labels)
    return torch.from_numpy(labels_arr[:, None] == labels_arr[None, :])


@torch.no_grad()
def evaluate(
    val_loader: DataLoader[Sample],
    model: Embedder,
    writer: SummaryWriter | None = None,
) -> EvaluationMetrics:
    sup_con_loss = SupConLoss()
    model.eval()

    total_loss = 0.0
    total_count = 0

    n = len(val_loader.dataset)
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

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        photos = batch["photo"].to(DEVICE, non_blocking=True)
        sketches = batch["sketch"].to(DEVICE, non_blocking=True)

        pe, se = model(photos, sketches)
        ids = to_ids(batch["categories"])
        mask = ids.unsqueeze(0) == ids.unsqueeze(1)

        loss = sup_con_loss(pairviews(torch.cat([pe, se], dim=0)), mask=mask)

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

    assert photo_embeddings is not None and sketch_embeddings is not None

    sims = sketch_embeddings @ photo_embeddings.t()  # [Q,G]

    metrics = compute_map_at_k_chunked(
        query_emb=sketch_embeddings,
        query_labels=sketch_category_ids,
        gallery_emb=photo_embeddings,
        gallery_labels=photo_category_ids,
        ks=(10, 30),
        chunk_size=128,
    )

    map10, p10 = metrics[10]
    map30, p30 = metrics[30]
    del sims
    if writer is not None:
        # moving to CPU to free GPU memory
        # projector does not need GPU
        photo_embeddings = photo_embeddings.cpu()
        sketch_embeddings = sketch_embeddings.cpu()
        log_projector_embeddings(
            writer,
            photo_embeddings,
            sketch_embeddings,
            photo_category_ids,
            sketch_category_ids,
            tag="eval/projector",
        )
    del photo_embeddings, sketch_embeddings, photo_category_ids, sketch_category_ids
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    return EvaluationMetrics(
        loss=val_loss,
        mean_average_precision_at_10=map10,
        precision_at_10=p10,
        mean_average_precision_at_30=map30,
        precision_at_30=p30,
    )


def _make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opt_name = CFG.training.optimizer
    lr = float(CFG.training.base_lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _amp_dtype() -> torch.dtype:
    return torch.bfloat16


def _amp_enabled() -> bool:
    return DEVICE.type == "cuda"


def train_phase1_dcl(
    model: Embedder,
    loader: DataLoader[Sample],
    writer: SummaryWriter,
    run_name: str,
    *,
    start_step: int = 0,
) -> int:
    """
    First phase is self-supervised DCL on paired instances.
    """
    epochs = int(CFG.training.phase_1.epochs)
    if epochs <= 0:
        return start_step

    optimizer = _make_optimizer(model)
    total_steps = epochs * len(loader)
    warmup_steps = CFG.training.phase_1.warmup_steps
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=CFG.training.min_lr_ratio,
    )

    temperature = float(CFG.training.phase_1.temperature)

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    global_step = start_step
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(
            loader,
            desc=f"[Phase1/DCL] Epoch {epoch + 1}/{epochs}",
            miniters=50,
        )
        for batch in pbar:
            photo = batch["photo"].to(DEVICE, non_blocking=True)
            sketch = batch["sketch"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=_amp_dtype(),
                enabled=_amp_enabled(),
            ):
                pe, se = model(photo, sketch)
                loss = dcl_loss(pe, se, temperature=temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (global_step + 1) % 50 == 0:
                writer.add_scalar(
                    "phase1/train_loss_dcl",
                    float(loss.detach()),
                    global_step,
                )
                writer.add_scalar(
                    "phase1/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )
                pbar.set_postfix({"loss": float(loss.detach())})

            global_step += 1

        out_dir = Path(CFG.training.model_save_path) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"phase1_epoch_{epoch + 1}.pth")
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return global_step


def train_phase2_supcon(
    model: Embedder,
    train_loader: DataLoader[Sample],
    val_loader: DataLoader[Sample],
    writer: SummaryWriter,
    run_name: str,
    *,
    start_step: int = 0,
) -> None:
    epochs = int(CFG.training.phase_2.epochs)
    optimizer = _make_optimizer(model)
    total_steps = epochs * len(train_loader)
    warmup_steps = CFG.training.phase_2.warmup_steps
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=CFG.training.min_lr_ratio,
    )

    sup_con_loss = SupConLoss(
        temperature=float(CFG.training.phase_2.temperature),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    eval_every = CFG.validation.eval_every_steps

    global_step = start_step
    end_step = start_step + epochs * len(train_loader)
    model.train()

    best_map10 = -1.0

    for epoch in range(epochs):
        pbar = tqdm(
            train_loader,
            desc=f"[Phase2/SupCon] Epoch {epoch + 1}/{epochs}",
            miniters=50,
        )
        for batch in pbar:
            photo = batch["photo"].to(DEVICE, non_blocking=True)
            sketch = batch["sketch"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=_amp_dtype(),
                enabled=_amp_enabled(),
            ):
                pe, se = model(photo, sketch)
                mask = _create_class_mask(batch["categories"]).to(DEVICE)
                loss = sup_con_loss(pairviews(torch.cat([pe, se], dim=0)), mask=mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (global_step + 1) % 50 == 0:
                writer.add_scalar(
                    "phase2/train_loss_supcon",
                    float(loss.detach()),
                    global_step,
                )
                writer.add_scalar(
                    "phase2/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )
                pbar.set_postfix({"loss": float(loss.detach())})

            is_last_step = global_step == end_step - 1
            do_eval = (
                (global_step + 1) % eval_every == 0
                or global_step % (len(train_loader) - 1) == 0
                or global_step == start_step
                or is_last_step
            )

            if do_eval:
                if is_last_step:
                    eval_metrics = evaluate(val_loader, model, writer)
                else:
                    eval_metrics = evaluate(val_loader, model)
                model.train()

                writer.add_scalar("eval/loss", eval_metrics.loss, global_step)
                writer.add_scalar(
                    "eval/mAP@10",
                    eval_metrics.mean_average_precision_at_10,
                    global_step,
                )
                writer.add_scalar(
                    "eval/Precision@10",
                    eval_metrics.precision_at_10,
                    global_step,
                )
                writer.add_scalar(
                    "eval/mAP@30",
                    eval_metrics.mean_average_precision_at_30,
                    global_step,
                )
                writer.add_scalar(
                    "eval/Precision@30",
                    eval_metrics.precision_at_30,
                    global_step,
                )

                tqdm.write(
                    f"[Eval step {global_step + 1}] "
                    f"loss={eval_metrics.loss:.4f} "
                    f"mAP@10={eval_metrics.mean_average_precision_at_10:.4f} "
                    f"P@10={eval_metrics.precision_at_10:.4f} "
                    f"mAP@30={eval_metrics.mean_average_precision_at_30:.4f} "
                    f"P@30={eval_metrics.precision_at_30:.4f}",
                )

                # best checkpoint by mAP@10
                if eval_metrics.mean_average_precision_at_10 > best_map10:
                    best_map10 = eval_metrics.mean_average_precision_at_10
                    out_dir = Path(CFG.training.model_save_path) / run_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out_dir / "best_phase2_by_map10.pth")

            global_step += 1

        out_dir = Path(CFG.training.model_save_path) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"phase2_epoch_{epoch + 1}.pth")


def train() -> None:
    encoder = TimmBackbone(name=CFG.skitter.encoder_name)
    encoder.to(DEVICE)
    encoder.eval()

    model = Embedder(
        backbone=encoder,
        hidden_layer_size=CFG.skitter.projection_head_size,
        embedding_size=CFG.skitter.embedding_size,
    ).to(DEVICE)

    model = torch.compile(model, mode="reduce-overhead")

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # =========
    # Phase 1: DCL on paired data (no train/val/test split)
    # =========
    phase1_epochs = int(CFG.training.phase_1.epochs)
    global_step = 0
    if phase1_epochs > 0:
        phase1_images = Path(
            CFG.training.phase_1.images_path,
        )
        phase1_sketches = Path(
            CFG.training.phase_1.sketches_path,
        )
        # ignored for now, time is running short :))
        _phase1_fraction = float(
            CFG.training.phase_1.fraction_of_samples,
        )

        phase1_samples = get_qmul_paired_samples(
            images_root=phase1_images,
            sketches_root=phase1_sketches,
        )
        phase1_loader = build_loader(
            samples=phase1_samples,
            use_class_balanced_sampler=False,
            batch_size=int(
                CFG.training.batch_size,
            ),
            shuffle=True,
            photo_transform=build_photo_transform_train(),
            sketch_transform=build_sketch_transform_train(),
            num_workers=CFG.training.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
        global_step = train_phase1_dcl(
            model,
            phase1_loader,
            writer,
            run_name,
            start_step=global_step,
        )

    # =========
    # Phase 2: SupCon on labeled data (train/val/test split)
    # =========
    train_samples, val_samples, _test_samples = get_samples_from_directories(
        images_root=Path(CFG.training.phase_2.images_path),
        sketches_root=Path(CFG.training.phase_2.sketches_path),
        per_category_fraction=CFG.training.phase_2.fraction_of_samples,
        val_fraction=CFG.validation.validation_fraction,
        test_fraction=CFG.training.test_fraction,
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

    train_phase2_supcon(
        model,
        train_loader,
        val_loader,
        writer,
        run_name,
        start_step=global_step,
    )

    writer.close()
