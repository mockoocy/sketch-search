from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from sktr.config.config import DEVICE

from sktr.type_defs import (
    Batch,
    GrayTorch,
    ImageTransformFunction,
    RGBTorch,
    Sample,
    SamplePath,
)
from collections import defaultdict

# formats supported by torchivision.io.read_image
IMG_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif"]


class PhotoSketchDataset(Dataset[Sample]):
    """Dataset for pairing photos and sketches.

    Args:
        samples: list of paths to a (image, sketch) pair.
    """

    def __init__(
        self,
        samples: list[SamplePath],
        sketch_as_rgb: bool = False,
        photo_transform: ImageTransformFunction | None = None,
        sketch_transform: ImageTransformFunction | None = None,
    ) -> None:
        self.samples = samples
        self.sketch_as_rgb = sketch_as_rgb
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

    def __len__(self) -> int:
        return len(self.samples)

    @torch.no_grad()
    def _load_photo(self, path: Path) -> RGBTorch:
        photo = io.read_image(str(path), mode=io.ImageReadMode.RGB).float() / 255.0
        return (
            self.photo_transform(photo) if self.photo_transform is not None else photo
        )

    @torch.no_grad()
    def _load_sketch(self, path: Path) -> GrayTorch:
        try:
            sketch = (
                io.read_image(str(path), mode=io.ImageReadMode.GRAY).float() / 255.0
            )
        except RuntimeError as e:
            print(f"Error loading image {path}: {e}")
            raise e
        if self.sketch_transform is None:
            return sketch.squeeze(0)  # [1, H, W] -> [H, W]
        # Averaging there may be not the best idea, but works for now.
        # It may distort values, since the transform may include
        # normalization, but it is ok for now.
        res = self.sketch_transform(sketch.repeat(3, 1, 1)).mean(
            dim=0
        )  # [3, H, W] -> [H, W]
        return res

    def __getitem__(self, idx: int) -> Sample:
        """Get one sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            A dictionary containing the photo and sketch tensors.
        """
        return Sample(
            photo=self._load_photo(self.samples[idx].photo),
            sketch=self._load_sketch(self.samples[idx].sketch),
            category=self.samples[idx].category,
        )


def _collate_fn(batch: list[Sample]) -> Batch:
    """Custom collate function for DataLoader."""
    return {
        "photo": torch.stack([item.photo for item in batch]),
        "sketch": torch.stack([item.sketch for item in batch]),
        "categories": [item.category for item in batch],
    }


def _worker_init(_):
    torch.set_num_threads(1)


def build_loader(
    samples: list[SamplePath],
    batch_size: int = 32,
    num_workers: int = 0,
    photo_transform: ImageTransformFunction | None = None,
    sketch_transform: ImageTransformFunction | None = None,
    shuffle: bool = True,  # noqa: FBT001, FBT002 - false positive
    sketch_as_rgb: bool = False,
    drop_last: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader[Sample]:
    """
    Returns a DataLoader ready for training/testing loops.
    """
    dataset = PhotoSketchDataset(
        samples,
        sketch_as_rgb=sketch_as_rgb,
        photo_transform=photo_transform,
        sketch_transform=sketch_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=_collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init,
    )


def get_samples_from_directories(
    images_root: Path,
    sketches_root: Path,
    per_category_fraction: float = 1.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[SamplePath], list[SamplePath], list[SamplePath]]:
    """Assembles collection of (image, sketch) pairs.

    These pairs are created with respect to the categories.
    It is assumed that the directories follow {root}/{category}/*
    file structure.

    The amount of samples per category is equal to number of sketches in
    a given category, multiplied by `per_category_fraction`.
    For each sample, a random photo is chosen from the photo category,
    which serves as a mechanism for mismatch in number of photos and sketches.

    Then the categories are shuffled and split into train/val/test sets,
    according to `val_fraction` and `test_fraction` parameters.
    The number of unseen categories (which are used both for val and test)
    is determined by these - not the number of samples.

    Args:
        images_root: Path to the root directory containing images.
        sketches_root: Path to the root directory containing sketches.
        per_category_fraction: Fraction of images to use per category.

    Returns:
        A tuple of three lists of (image, sketch) path pairs:
        (train_samples, val_samples, test_samples)
    """
    all_samples: defaultdict[str, list[SamplePath]] = defaultdict(list)
    random.seed(seed)
    print(
        f"directories (absolute): images: {images_root.resolve()}, sketches: {sketches_root.resolve()}"
    )
    category_dirs = [
        directory for directory in images_root.iterdir() if directory.is_dir()
    ]
    for category_dir in category_dirs:
        image_files = list(
            file
            for file in (category_dir).glob("*")
            if file.suffix[1:].lower() in IMG_EXTENSIONS
        )
        sketch_category = sketches_root / category_dir.name
        sketch_files = list(
            file
            for file in sketch_category.glob("*")
            if file.suffix[1:].lower() in IMG_EXTENSIONS
        )
        number_of_samples = int(per_category_fraction * len(sketch_files))
        for sketch_path, image_path in zip(
            sketch_files, random.choices(image_files, k=number_of_samples)
        ):
            all_samples[category_dir.name].append(
                SamplePath(
                    photo=image_path,
                    sketch=sketch_path,
                    category=category_dir.name,
                )
            )

    random.shuffle(category_dirs)
    unseen_classes_count = int(len(category_dirs) * (val_fraction + test_fraction))
    unseen_classes = set(
        category_dir.name for category_dir in category_dirs[:unseen_classes_count]
    )
    unseen_test_fraction = (
        test_fraction / (val_fraction + test_fraction)
        if (val_fraction + test_fraction) > 0
        else 0.0
    )

    train_samples: list[SamplePath] = []
    val_samples: list[SamplePath] = []
    test_samples: list[SamplePath] = []
    for category_dir, samples in all_samples.items():
        if category_dir not in unseen_classes:
            train_samples.extend(samples)
            continue
        random.shuffle(samples)
        split_idx = int(len(samples) * unseen_test_fraction)
        test_samples.extend(samples[:split_idx])
        val_samples.extend(samples[split_idx:])
    return train_samples, val_samples, test_samples
