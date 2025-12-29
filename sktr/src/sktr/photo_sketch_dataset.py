import random
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import io

from sktr.type_defs import (
    Batch,
    GrayTorch,
    ImageTransformFunction,
    RGBTorch,
    Sample,
    SamplePath,
)

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
        photo_transform: ImageTransformFunction | None = None,
        sketch_transform: ImageTransformFunction | None = None,
        *,
        sketch_as_rgb: bool = False,
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
        sketch = io.read_image(str(path), mode=io.ImageReadMode.GRAY).float() / 255.0
        if self.sketch_transform is None:
            return sketch.squeeze(0)  # [1, H, W] -> [H, W]
        # Averaging there may be not the best idea, but works for now.
        # It may distort values, since the transform may include
        # normalization, but it is ok for now.
        return self.sketch_transform(sketch.repeat(3, 1, 1)).mean(
            dim=0,
        )  # [3, H, W] -> [H, W]

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


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


class ClassBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        samples: list[SamplePath],
        classes_per_batch: int,
        samples_per_class: int,
        *,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.by_class: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            self.by_class[s.category].append(i)

        self.classes = list(self.by_class.keys())
        self.num_classes = len(self.classes)

        self.classes_per_batch = min(self.classes_per_batch, self.num_classes)
        self.batch_size = self.classes_per_batch * self.samples_per_class

        total = len(samples)
        self.num_batches = total // self.batch_size if drop_last else np.ceil(
            total / self.batch_size
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = self.rng.sample(self.classes, self.classes_per_batch)

            batch: list[int] = []

            for c in chosen_classes:
                indices = self.by_class[c]

                if len(indices) >= self.samples_per_class:
                    picked = self.rng.sample(indices, self.samples_per_class)
                else:
                    # sample with replacement if class is small
                    picked = self.rng.choices(indices, k=self.samples_per_class)
                batch.extend(picked)

            self.rng.shuffle(batch)
            yield batch



def build_loader(  # noqa: PLR0913
    samples: list[SamplePath],
    *,
    use_class_balanced_sampler: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    photo_transform: ImageTransformFunction | None = None,
    sketch_transform: ImageTransformFunction | None = None,
    prefetch_factor: int = 4,
    shuffle: bool = True,
    sketch_as_rgb: bool = False,
    drop_last: bool = True,
    persistent_workers: bool = True,
    samples_per_class: int = 4,
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
        batch_size=batch_size if not use_class_balanced_sampler else 1,
        batch_sampler=ClassBalancedBatchSampler(
            samples,
            classes_per_batch=max(1, batch_size // samples_per_class),
            samples_per_class=samples_per_class,
            drop_last=drop_last,
        )
        if use_class_balanced_sampler
        else None,
        shuffle=shuffle if not use_class_balanced_sampler else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last if not use_class_balanced_sampler else False,
        collate_fn=_collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init,
    )


def get_samples_from_directories(  # noqa: PLR0913
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
    category_dirs = [
        directory for directory in images_root.iterdir() if directory.is_dir()
    ]
    for category_dir in category_dirs:
        image_files = [
            file
            for file in (category_dir).glob("*")
            if file.suffix[1:].lower() in IMG_EXTENSIONS
        ]
        sketch_category = sketches_root / category_dir.name
        sketch_files = [
            file
            for file in sketch_category.glob("*")
            if file.suffix[1:].lower() in IMG_EXTENSIONS
        ]
        number_of_samples = max(4, int(per_category_fraction * len(sketch_files)))
        for sketch_path, image_path in zip(
            sketch_files,
            random.choices(image_files, k=number_of_samples),  # noqa: S311 - not a cryptographic use
            strict=False,
        ):
            all_samples[category_dir.name].append(
                SamplePath(
                    photo=image_path,
                    sketch=sketch_path,
                    category=category_dir.name,
                ),
            )

    random.shuffle(category_dirs)
    unseen_classes_count = int(len(category_dirs) * (val_fraction + test_fraction))
    unseen_classes = {
        category_dir.name for category_dir in category_dirs[:unseen_classes_count]
    }
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


def get_qmul_paired_samples(
    images_root: Path,
    sketches_root: Path,
) -> list[SamplePath]:
    """
    QMUL FG-SBIR pairing:
    - photo: <instance_id>.png
    - sketches: <instance_id>_<k>.png
    """
    photos = {
        p.stem: p
        for p in images_root.iterdir()
        if p.suffix[1:].lower() in IMG_EXTENSIONS
    }

    paired = list[SamplePath]()

    for sk in sketches_root.iterdir():
        format = sk.suffix[1:].lower()
        if format not in IMG_EXTENSIONS:
            continue

        stem = sk.stem
        if "_" not in stem:
            continue

        instance_id = stem.rsplit("_", 1)[0]

        photo = photos.get(instance_id)
        if photo is None:
            continue
        paired.append(
            SamplePath(
                photo=photo,
                sketch=sk,
                category=instance_id,  # instance-level label
            )
        )

    return paired
