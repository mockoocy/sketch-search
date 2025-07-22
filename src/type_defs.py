from typing import Literal, TypedDict

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


class Sample(TypedDict):
    photo: RGBTorch
    sketch: GrayTorch


class Batch(TypedDict):
    photo: RGBTorchBatch
    sketch: GrayTorchBatch


class JigsawBatch(Batch):
    puzzle: RGBTorchBatch
    source_mask: SourceMaskBatch
