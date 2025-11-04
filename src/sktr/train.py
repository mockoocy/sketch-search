from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
import torch
from sktr.config.config import CFG, DEVICE
from sktr.model import SKTR, SupConLoss, TimmBackbone, two_way_dcl_loss
from sktr.photo_sketch_dataset import build_loader, get_samples_from_directories
from sktr.type_defs import ImageTransformFunction, Sample
from tqdm import tqdm
import timm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sktr.vector import EvaluationStore


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


@torch.no_grad()
def evaluate(
    val_loader: DataLoader[Sample], model: SKTR, global_step: int = 0
) -> EvaluationMetrics:
    total_loss = 0.0
    count = 0
    eval_store = EvaluationStore(
        embedding_size=CFG.skitter.embedding_size,
    )
    sup_con_loss = SupConLoss()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            photos = batch["photo"].to(DEVICE)
            sketches = batch["sketch"].to(DEVICE)
            batch_size = photos.size(0)
            photo_embeddings, sketch_embeddings = model(photos, sketches)
            eval_store.upsert(
                embeddings=photo_embeddings.to("cpu"),
                paths=[
                    f"photo_{global_step}_{photo_idx}"
                    for photo_idx in range(batch_size)
                ],
                categories=batch["categories"],
            )
            mask = _create_class_mask(batch["categories"]).to(DEVICE)
            loss = sup_con_loss(
                pairviews(
                    torch.cat([photo_embeddings, sketch_embeddings], dim=0).to(DEVICE)
                ),
                mask=mask,
            )
            total_loss += loss.item() * photos.size(0)
            count += photos.size(0)
        val_loss = total_loss / count if count > 0 else float("inf")
        # second pass for retrieval metrics
        map_at_10 = 0.0
        total_sample_count = 0
        for batch in tqdm(val_loader, desc="Evaluating retrieval"):
            photos = batch["photo"].to(DEVICE)
            sketches = batch["sketch"].to(DEVICE)
            batch_size = photos.size(0)
            sketch_embeddings = model.embed_sketch(sketches)
            total_sample_count += batch_size
            map_at_10 += (
                eval_store.mean_average_precision_at_k(
                    query_embeddings=sketch_embeddings.to("cpu"),
                    query_categories=batch["categories"],
                    top_k=10,
                )
                * batch_size
            )
        # There is only function for mAP calculation
        # So mAP is taken as a weighted average of
        # smaller mAPs :)
        # Weighted instead of arithmetic, due to drop_last=False
        # Thus batch may be smaller than batch_size
        map_at_10 /= total_sample_count
    return EvaluationMetrics(
        loss=val_loss,
        mean_average_precision_at_10=map_at_10,
    )


def train():
    print(f"Training on {DEVICE=}")
    encoder = TimmBackbone(
        name=CFG.skitter.encoder_name,
    )
    encoder.to(DEVICE)
    encoder.eval()
    model = SKTR(
        encoder=encoder,
        hidden_layer_size=CFG.skitter.projection_head_size,
        embedding_size=CFG.skitter.embedding_size,
    )
    model.to(DEVICE)
    model = torch.compile(model, mode="reduce-overhead")

    cfg = timm.data.resolve_data_config(pretrained_cfg=encoder.model.pretrained_cfg)
    transform: ImageTransformFunction = timm.data.create_transform(
        **cfg,
        is_training=True,
    )
    transform_eval: ImageTransformFunction = timm.data.create_transform(
        **cfg,
        is_training=False,
    )

    train_samples, val_samples, test_samples = get_samples_from_directories(
        images_root=Path(CFG.training.images_path),
        sketches_root=Path(CFG.training.sketches_path),
        per_category_fraction=CFG.training.fraction_of_samples,
        val_fraction=CFG.validation.validation_fraction,
        test_fraction=CFG.training.test_fraction,
    )

    train_loader = build_loader(
        train_samples,
        batch_size=CFG.training.batch_size,
        photo_transform=transform,
        sketch_transform=transform,
        num_workers=6,
    )
    val_loader: DataLoader[Sample] = build_loader(
        samples=val_samples,
        batch_size=CFG.training.batch_size,
        photo_transform=transform_eval,
        sketch_transform=transform_eval,
        shuffle=False,
        drop_last=False,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=False,
    )
    sup_con_loss = SupConLoss()

    if CFG.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.training.base_lr, weight_decay=0.05, fused=True
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.training.base_lr)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * CFG.training.epochs
    warmup_steps = int(0.05 * total_steps)
    lr_scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=0.1
    )
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for epoch in range(epochs := CFG.training.epochs):
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", miniters=50
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
                writer.add_scalar(
                    "train/lr", lr_scheduler.get_last_lr()[0], global_step_number
                )
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # When the scale decreases, the optimizer step in `scaler.step` is skipped
            # Thus we also skip the lr_scheduler step
            skip_lr_sched = scale_before > scaler.get_scale()
            if not skip_lr_sched:
                lr_scheduler.step()
            if (global_step_number + 1) % 50 == 0:
                progress_bar.set_postfix({"loss": float(loss.detach())})
            is_last_step = batch_index + 1 == len(train_loader)
            if (
                global_step_number + 1
            ) % CFG.validation.eval_every_steps == 0 or is_last_step:
                eval_metrics = evaluate(
                    val_loader, model, global_step=global_step_number
                )
                writer.add_scalar("eval/loss", eval_metrics.loss, global_step_number)
                writer.add_scalar(
                    "eval/mAP@10",
                    eval_metrics.mean_average_precision_at_10,
                    global_step_number,
                )
                tqdm.write(
                    f"Eval at epoch {epoch + 1}, step {global_step_number + 1}: "
                    f"loss={eval_metrics.loss:.4f}, "
                    f"mAP@10={eval_metrics.mean_average_precision_at_10:.4f}, "
                )
        model_save_path = (
            Path(CFG.training.model_save_path) / f"sktr_model_epoch_{epoch + 1}.pth"
        )
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
