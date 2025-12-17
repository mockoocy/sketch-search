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

from sktr.config.config import CFG, DEVICE
from sktr.model import (
    Embedder,
    SupConLoss,
    TimmBackbone,
    build_photo_transform_eval,
    build_photo_transform_train,
    build_sketch_transform_eval,
    build_sketch_transform_train,
)
from sktr.photo_sketch_dataset import build_loader, get_samples_from_directories
from sktr.type_defs import Sample

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


@dataclass
class EvaluationMetrics:
    loss: float
    mean_average_precision_at_10: float


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,  # final LR = base_lr * min_lr_ratio
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        temperature = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * temperature))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pairviews(x: torch.Tensor) -> torch.Tensor:
    """Small helper to make inputs to SupConLoss

    Args:
        x: Tensor to resize

    Returns:
        Resizes tensor from shape [2*B, ...] -> [B, 2, ...]
    """
    batch_size = x.shape[0] // 2
    return x.reshape(2, batch_size, *x.shape[1:]).transpose(0, 1).contiguous()


@torch.no_grad()
def _create_class_mask(labels: list[str]) -> torch.Tensor:
    labels_arr = np.array(labels)
    class_mask = labels_arr[:, None] == labels_arr[None, :]
    return torch.from_numpy(class_mask)


def train() -> None:
    encoder = TimmBackbone(
        name=CFG.skitter.encoder_name,
    )
    encoder.to(DEVICE)
    encoder.eval()
    model = Embedder(
        backbone=encoder,
        hidden_layer_size=CFG.skitter.projection_head_size,
        embedding_size=CFG.skitter.embedding_size,
    )
    model.to(DEVICE)
    model.train()
    model = torch.compile(model, mode="reduce-overhead")

    train_samples, val_samples, _test_samples = get_samples_from_directories(
        images_root=Path(CFG.training.images_path),
        sketches_root=Path(CFG.training.sketches_path),
        per_category_fraction=CFG.training.fraction_of_samples,
        val_fraction=CFG.validation.validation_fraction,
        test_fraction=CFG.training.test_fraction,
    )

    train_loader = build_loader(
        train_samples,
        batch_size=CFG.training.batch_size,
        photo_transform=build_photo_transform_train(),
        sketch_transform=build_sketch_transform_train(),
        num_workers=6,
    )
    val_loader: DataLoader[Sample] = build_loader(
        samples=val_samples,
        use_class_balanced_sampler=False,
        batch_size=CFG.training.batch_size,
        photo_transform=build_photo_transform_eval(),
        sketch_transform=build_sketch_transform_eval(),
        drop_last=False,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=False,
    )

    sup_con_loss = SupConLoss()

    if CFG.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.training.base_lr,
            weight_decay=0.05,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.training.base_lr)
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for epoch in range(epochs := CFG.training.epochs):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            miniters=50,
        )
        for batch_index, batch in enumerate(progress_bar):
            global_step_number = epoch * len(train_loader) + batch_index
            global_step_number = epoch * len(train_loader) + batch_index
            photo = batch["photo"].to(DEVICE, non_blocking=True)
            sketch = batch["sketch"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                photo_embedding, sketch_embedding = model(photo, sketch)
                mask = _create_class_mask(batch["categories"]).to(DEVICE)
                loss = sup_con_loss(
                    pairviews(torch.cat([photo_embedding, sketch_embedding], dim=0)),
                    mask=mask,
                )

            if (global_step_number + 1) % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step_number)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (global_step_number + 1) % 50 == 0:
                progress_bar.set_postfix({"loss": float(loss.detach())})
            is_last_step = batch_index + 1 == len(train_loader)
            if (
                (global_step_number + 1) % CFG.validation.eval_every_steps == 0
                or is_last_step
                or global_step_number == 0
            ):
                model.eval()
                eval_metrics = evaluate(
                    val_loader,
                    model,
                )
                model.train()
                writer.add_scalar("eval/loss", eval_metrics.loss, global_step_number)
                writer.add_scalar(
                    "eval/mAP@10",
                    eval_metrics.mean_average_precision_at_10,
                    global_step_number,
                )
                tqdm.write(
                    f"Eval at epoch {epoch + 1}, step {global_step_number + 1}: "
                    f"loss={eval_metrics.loss:.4f}, "
                    f"mAP@10={eval_metrics.mean_average_precision_at_10:.4f}, ",
                )
        model_save_path = (
            Path(CFG.training.model_save_path) / f"sktr_model_epoch_{epoch + 1}.pth"
        )
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)


@torch.no_grad()
def evaluate(val_loader: DataLoader[Sample], model: Embedder) -> EvaluationMetrics:  # noqa: PLR0915
    sup_con_loss = SupConLoss()
    total_loss = 0.0
    count = 0
    n = len(val_loader.dataset)
    photo_embeddings = None
    sketch_embeddings = None
    photo_category_ids = torch.empty(n, dtype=torch.uint8, device=DEVICE)
    sketch_category_ids = torch.empty(n, dtype=torch.uint8, device=DEVICE)
    label_to_id = {}
    idx = 0

    def to_ids(names: list[str]) -> torch.Tensor:
        ids = []
        for s in names:
            if s not in label_to_id:
                label_to_id[s] = len(label_to_id)
            ids.append(label_to_id[s])
        return torch.tensor(ids, device=DEVICE, dtype=torch.long)

    for batch in tqdm(val_loader, desc="Evaluating"):
        photos = batch["photo"].to(DEVICE, non_blocking=True)
        sketches = batch["sketch"].to(DEVICE, non_blocking=True)
        pe, se = model(photos, sketches)
        ids = to_ids(batch["categories"])
        mask = ids.unsqueeze(0) == ids.unsqueeze(1)
        loss = sup_con_loss(pairviews(torch.cat([pe, se], dim=0)), mask=mask)
        bsz = pe.size(0)
        total_loss += float(loss) * bsz
        count += bsz
        if photo_embeddings is None:
            d = pe.size(1)
            photo_embeddings = torch.empty(n, d, device=DEVICE, dtype=pe.dtype)
            sketch_embeddings = torch.empty(n, d, device=DEVICE, dtype=se.dtype)
        photo_embeddings[idx : idx + bsz].copy_(pe)
        sketch_embeddings[idx : idx + bsz].copy_(se)
        photo_category_ids[idx : idx + bsz] = ids
        sketch_category_ids[idx : idx + bsz] = ids
        idx += bsz

    val_loss = total_loss / count if count > 0 else float("inf")

    photo_embeddings = nn.functional.normalize(photo_embeddings, dim=1)
    sketch_embeddings = nn.functional.normalize(sketch_embeddings, dim=1)
    sims = sketch_embeddings @ photo_embeddings.t()
    k = 10
    topk = torch.topk(sims, k=k, dim=1).indices
    rel = sketch_category_ids.unsqueeze(1) == photo_category_ids.unsqueeze(0)
    rel_at_k = torch.gather(rel, 1, topk)
    csum = torch.cumsum(rel_at_k.int(), dim=1)
    ranks = torch.arange(1, k + 1, device=DEVICE).unsqueeze(0)
    prec = csum.float() / ranks.float()
    ap_num = (prec * rel_at_k.float()).sum(dim=1)
    denom = torch.minimum(rel.sum(dim=1).float(), torch.tensor(float(k), device=DEVICE))
    denom = torch.clamp_min(denom, 1.0)
    map_at_10 = (ap_num / denom).mean().item()

    # It's criminal, but we have to do this to free GPU memory
    del photo_embeddings, sketch_embeddings, photo_category_ids, sketch_category_ids
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    return EvaluationMetrics(loss=val_loss, mean_average_precision_at_10=map_at_10)
