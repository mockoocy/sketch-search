from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float, Int

type GrayU8NumpyBatch = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type RGBU8NumpyBatch = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]

# F722 is saying that the strings in the type aliases are giving 'syntax errors'
# These are there only to provide a human-readable shape description
type GrayTorch = Float[torch.Tensor, "Height Width"]  # noqa: F722
type GrayTorchBatch = Float[torch.Tensor, "Batch Height Width"]  # noqa: F722
type RGBTorch = Float[torch.Tensor, "3 Height Width"]  # noqa: F722
type RGBTorchBatch = Float[torch.Tensor, "Batch 3 Height Width"]  # noqa: F722
type SourceMaskBatch = Int[torch.Tensor, "Batch SourceMask"]  # noqa: F722

type Embedding = Float[torch.Tensor, "EmbeddingDimension"]  # noqa: F722
type EmbeddingBatch = Float[torch.Tensor, "Batch EmbeddingDimension"]  # noqa: F722

type Loss = Float[torch.Tensor, "0"]

type ImageType = Literal["Image", "Sketch"]

type StepFunction = Callable[[Batch], Loss]

type ImageTransformFunction = Callable[[RGBTorch | GrayTorch], RGBTorch | GrayTorch]


@dataclass(frozen=True)
class SamplePath:
    photo: Path
    sketch: Path
    category: str


@dataclass(frozen=True)
class Sample:
    photo: RGBTorch
    sketch: GrayTorch
    category: str


class Batch(TypedDict):
    photo: RGBTorchBatch
    sketch: GrayTorchBatch
    categories: list[str]


class StoredImage(TypedDict):
    path: str
    vector: list[float]
    category: str
