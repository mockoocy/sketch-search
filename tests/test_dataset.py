from torch.utils.data import DataLoader

from src.type_defs import Sample
from tests.conftest import VisualizeFunction


def test_loader_batch_shapes(
    dataloader: DataLoader[Sample],
    visualize: VisualizeFunction | None,
) -> None:
    """Ensure DataLoader returns correctly shaped batches."""
    batch = next(iter(dataloader))

    assert batch["photo"].ndim == 4  # [B,3,H,W]
    assert batch["sketch"].ndim == 3  # [B,H,W]
    batch_size = batch["photo"].shape[0]

    if dataloader.dataset.use_jigsaw:
        assert batch["puzzle"].shape == batch["photo"].shape
        assert batch["source_mask"].shape == (batch_size, 9)
    else:
        assert "puzzle" not in batch
        assert "source_mask" not in batch

    if visualize:
        sketches_rgb = batch["sketch"].unsqueeze(1).repeat(1, 3, 1, 1)  # [B,3,H,W]
        if dataloader.dataset.use_jigsaw:
            visualize(sketches_rgb, batch["photo"], batch["puzzle"])
        else:
            visualize(sketches_rgb, batch["photo"])
