from torch.utils.data import DataLoader

from sktr.type_defs import Sample
from tests.conftest import VisualizeFunction


def test_loader_batch_shapes(
    dataloader: DataLoader[Sample],
    visualize: VisualizeFunction | None,
) -> None:
    """Ensure DataLoader returns correctly shaped batches."""
    batch = next(iter(dataloader))

    assert batch["photo"].ndim == 4  # [B,3,H,W]
    assert batch["sketch"].ndim == 3  # [B,H,W]

    assert "puzzle" not in batch
    assert "source_mask" not in batch

    if visualize:
        sketches_rgb = batch["sketch"].unsqueeze(1).repeat(1, 3, 1, 1)  # [B,3,H,W]
        visualize(sketches_rgb, batch["photo"])
