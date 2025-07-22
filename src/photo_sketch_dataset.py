from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import torchvision.transforms.v2 as T  # noqa: N812 - that's a common convention
from torch.utils.data import DataLoader, Dataset
from torchvision import io

from src.config import CFG
from src.processing import make_jigsaw_puzzle
from src.type_defs import Batch, GrayTorch, JigsawBatch, RGBTorch, Sample

_resizer = T.Resize((CFG.image_size, CFG.image_size))


class PhotoSketchDataset(Dataset[Sample]):
    def __init__(
        self,
        image_paths: list[Path],
        sketch_paths: list[Path],
        use_jigsaw: bool = False,  # noqa: FBT001, FBT002 - false positive
        grid_side: int = 3,
    ) -> None:
        self.image_paths = image_paths
        self.sketch_paths = sketch_paths
        self.use_jigsaw = use_jigsaw
        self.grid_side = grid_side

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_photo(self, path: Path) -> RGBTorch:
        img = io.read_image(str(path), mode=io.ImageReadMode.RGB).float() / 255.0
        return _resizer(img)  # Resize to 224x224
        return img  # [3, H, W]

    def _load_sketch(self, path: Path) -> GrayTorch:
        img = io.read_image(str(path), mode=io.ImageReadMode.GRAY).float() / 255.0
        img = _resizer(img)  # Resize to 224x224
        return img.squeeze(0)  # [1, H, W]

    def __getitem__(self, idx: int) -> Sample:
        return {
            "photo": self._load_photo(self.image_paths[idx]),
            "sketch": self._load_sketch(self.sketch_paths[idx]),
        }


def make_sketch_collate(
    grid_side: int = 3,
    use_jigsaw: bool = False,  # noqa: FBT001, FBT002 - false positive
) -> Callable[[list[Sample]], Batch]:
    """Returns a closure capturing config flags."""

    def _collate(samples: list[Sample]) -> Batch:
        photos = torch.stack([sample["photo"] for sample in samples], 0)  # [B,3,H,W]
        sketches = torch.stack([sample["sketch"] for sample in samples], 0)  # [B,H,W]
        batch: Batch = {"photo": photos, "sketch": sketches}
        if use_jigsaw:
            puzzle, mask = make_jigsaw_puzzle(
                photos,  # (B,3,H,W)
                sketches,  # (B,H,W)
                grid_side=grid_side,
            )
            batch = cast("JigsawBatch", batch)
            batch.update({"puzzle": puzzle, "source_mask": mask})
        return batch

    return _collate


def build_loader(  # noqa: PLR0913 - it is what it is :/
    image_paths: list[Path],
    sketch_paths: list[Path],
    batch_size: int = 32,
    num_workers: int = 0,
    grid_side: int = 3,
    shuffle: bool = True,  # noqa: FBT001, FBT002 - false positive
    pin_memory: bool = False,  # noqa: FBT001, FBT002 - false positive
    use_jigsaw: bool = False,  # noqa: FBT001, FBT002 - false positive
) -> DataLoader[Sample]:
    """
    Returns a DataLoader ready for training/testing loops.
    """
    dataset = PhotoSketchDataset(
        image_paths=[Path(path) for path in image_paths],
        sketch_paths=[Path(path) for path in sketch_paths],
        use_jigsaw=use_jigsaw,
        grid_side=grid_side,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=make_sketch_collate(use_jigsaw, grid_side),
        drop_last=False,
    )
