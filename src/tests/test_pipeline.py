from sktr.model import SKTR, dcl_loss, TimmBackbone
import torch

import torch.nn.functional as fun
import numpy as np
from torch.utils.data import DataLoader

from sktr.type_defs import Batch, Sample

def test_dcl_loss():
    photo_embeddings = torch.randn(8, 512)
    sketch_embeddings = torch.randn(8, 512)
    loss = dcl_loss(photo_embeddings, sketch_embeddings)
    assert loss.item() >= 0
    assert loss.ndim == 0


def test_toy_dcl_loss():
    temperature = 0.2
    photo_embedding_1 = fun.normalize(torch.tensor([0.4, 0.9]), dim=0)
    photo_embedding_2 = fun.normalize(torch.tensor([0.23, 0.12]), dim=0)
    sketch_embedding_1 = fun.normalize(torch.tensor([0.234, 0.234]), dim=0)
    sketch_embedding_2 = fun.normalize(torch.tensor([0.98, 0.12]), dim=0)

    photo_embeddings = torch.stack([photo_embedding_1, photo_embedding_2])
    sketch_embeddings = torch.stack([sketch_embedding_1, sketch_embedding_2])

    # First batch, positive term
    photo1_sketch1_similarity = torch.dot(photo_embedding_1, sketch_embedding_1)
    # First batch, negative term
    photo1_sketch2_similarity = torch.dot(photo_embedding_1, sketch_embedding_2)
    # First batch, negative term
    photo1_photo2_similarity = torch.dot(photo_embedding_1, photo_embedding_2)
    # First batch, negative term
    sketch1_photo2_similarity = torch.dot(sketch_embedding_1, photo_embedding_2)
    # First batch, negative term
    sketch1_sketch2_similarity = torch.dot(sketch_embedding_1, sketch_embedding_2)

    # Second batch, positive term
    photo2_sketch2_similarity = torch.dot(photo_embedding_2, sketch_embedding_2)
    # Second batch, negative term
    photo2_sketch1_similarity = torch.dot(photo_embedding_2, sketch_embedding_1)
    # Second batch, negative term
    photo2_photo1_similarity = torch.dot(photo_embedding_2, photo_embedding_1)
    # Second batch, negative term
    sketch2_photo1_similarity = torch.dot(sketch_embedding_2, photo_embedding_1)
    # Second batch, negative term
    sketch2_sketch1_similarity = torch.dot(sketch_embedding_2, sketch_embedding_1)

    expected_positive_loss = (
        -(photo1_sketch1_similarity + photo2_sketch2_similarity) / temperature
    )
    expected_negative_loss_batch_1 = torch.log(
        torch.exp(photo1_sketch2_similarity / temperature)
        + torch.exp(photo1_photo2_similarity / temperature)
        + torch.exp(sketch1_photo2_similarity / temperature)
        + torch.exp(sketch1_sketch2_similarity / temperature)
    )
    expected_negative_loss_batch_2 = torch.log(
        torch.exp(photo2_sketch1_similarity / temperature)
        + torch.exp(photo2_photo1_similarity / temperature)
        + torch.exp(sketch2_photo1_similarity / temperature)
        + torch.exp(sketch2_sketch1_similarity / temperature)
    )

    exptected_loss = (
        expected_positive_loss
        + (expected_negative_loss_batch_1 + expected_negative_loss_batch_2)
    ) / 2  # Average over the two batches

    loss = dcl_loss(photo_embeddings, sketch_embeddings, temperature=temperature)
    assert torch.isclose(loss, exptected_loss, atol=1e-6)


def test_gradients_exist_on_head(dataloader: DataLoader[Sample]) -> None:
    encoder = TimmBackbone(name="resnet18", pretrained=False)
    model = SKTR(encoder=encoder, hidden_layer_size=128, embedding_size=32)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    batch: Batch = next(iter(dataloader))
    photos = batch["photo"]
    sketches = batch["sketch"].unsqueeze(1).repeat(1, 3, 1, 1)

    p_emb, s_emb = model(photos, sketches)
    loss = dcl_loss(p_emb, s_emb, temperature=0.2)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    # at least one linear layer in the head should have grads
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in model.projection_head.named_parameters()
    )
    assert has_grad


def test_cpu_forward_and_train_smoke(dataloader: DataLoader[Sample]) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)

    # not pretrained to avoid internet
    encoder = TimmBackbone(name="resnet18", pretrained=False)
    model = SKTR(encoder=encoder, hidden_layer_size=256, embedding_size=64)

    batch: Batch = next(iter(dataloader))
    photos = batch["photo"]  # [B,3,H,W]
    sketches = (
        batch["sketch"].unsqueeze(1).repeat(1, 3, 1, 1)
    )  # [B,3,H,W] for the encoder

    with torch.no_grad():
        p_emb, s_emb = model(photos, sketches)
        assert p_emb.ndim == 2 and s_emb.ndim == 2
        assert p_emb.shape[0] == photos.shape[0] and s_emb.shape[0] == photos.shape[0]
        loss = dcl_loss(p_emb, s_emb, temperature=0.2)
        assert torch.isfinite(loss).item()

    iters = 3
    total_loss = 0.0
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= iters:
            break
        photos = batch["photo"]  # [B,3,H,W]
        sketches = batch["sketch"]
        photo_embeddings, sketch_embeddings = model(
            photos, sketches
        )
        loss = dcl_loss(photo_embeddings, sketch_embeddings, temperature=0.2)
        assert torch.isfinite(loss).item()
        total_loss += float(loss.detach().cpu())

    # loss should be a real number after a few steps
    assert np.isfinite(total_loss)
