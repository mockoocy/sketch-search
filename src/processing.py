import torch
import torch.nn.functional as F  # noqa: N812 - widely recognized convention

from src.type_defs import GrayTorchBatch, RGBTorchBatch, SourceMaskBatch

# fmt: off
_G123_LUT = torch.tensor(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    0, 1, 1, 0, 0, 1, 0, 0, 0],
    dtype=torch.bool)

_G123P_LUT = torch.tensor(
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=torch.bool)
# fmt: on

THINNING_MASK = torch.tensor(
    [[8, 4, 2], [16, 0, 1], [32, 64, 128]],
    dtype=torch.float32,
)

THINNING_CONV_KERNEL = THINNING_MASK.view(1, 1, 3, 3)


@torch.no_grad()
def binarize_by_percentile(
    edge_map: GrayTorchBatch,
    percentile: float = 95.0,
) -> GrayTorchBatch:
    """Binarizes an edge map by a given percentile.

    Args:
        edge_map: Input edge map tensor of shape (H,W) or (B,H,W).
        p: Percentile value for binarization. Defaults to 95.0.

    Returns:
        A binary tensor indicating edges.
    """
    flat = edge_map.flatten(start_dim=1)
    thresh = torch.quantile(flat.float(), percentile / 100)
    return (edge_map >= thresh).float()


@torch.no_grad()
def binary_thinning(
    binary: GrayTorchBatch,
    max_iter: int | None = None,
) -> GrayTorchBatch:
    device = binary.device
    kernel = THINNING_CONV_KERNEL.to(device)
    lut1, lut2 = _G123_LUT.to(device), _G123P_LUT.to(device)

    # Skeleton will be [B, 1, H, W] float with values {0, 1}.
    skeleton = binary.clone().unsqueeze(1).float()

    max_iter: float | int = max_iter if max_iter is not None else float("inf")

    curr_iter = 0
    while curr_iter < max_iter:
        changed = False
        for lut in (lut1, lut2):
            codes = F.conv2d(skeleton, kernel, padding=1)
            delete = lut[codes.int()]
            if delete.any():
                changed = True
                skeleton[delete] = 0.0
        if not changed:
            break
        curr_iter += 1
    return skeleton.squeeze(1).float()


@torch.no_grad()
def make_jigsaw_puzzle(
    photo: RGBTorchBatch,
    sketch: GrayTorchBatch,
    grid_side: int = 3,
    generator: torch.Generator | None = None,
) -> tuple[RGBTorchBatch, SourceMaskBatch]:
    """Build a mixed-modal jigsaw puzzle à la Pang et al. 2020.

    Each of the g*g cells is randomly filled with either a photo patch
    or the corresponding sketch patch (50 % probability).
    The g*g cells are then *independently* permuted for every sample.
    Works batch-wise, fully on GPU, no Python loops over pixels.

    Args:
        photo: Photo tensor of shape (B, 3, H, W) with RGB values.
        sketch: Sketch tensor of shape (B, 1, H, W) with binary or greyscale values.
        grid_side: Number of cells per side of the grid. Defaults to 3.
        generator: Optional random number generator for reproducibility.

    Returns:
        The jigsaw puzzle tensor of shape (B, 3, H, W) and a source mask -
        tensor of shape (B, g*g) indicating which cells are from the sketch.
    """
    batched = sketch.ndim == 3 and photo.ndim == 4  # Batched case
    if not batched:  # Unbatched case
        sketch = sketch.unsqueeze(0)  # [1, H, W]
        photo = photo.unsqueeze(0)  # [1, 3, H, W]
    sketch_rgb = sketch.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
    batch_size, _, height, width = photo.shape

    cell_height, cell_width = height // grid_side, width // grid_side

    photo_patches = F.unfold(
        photo.float(),
        (cell_height, cell_width),
        stride=(cell_height, cell_width),
    )
    sketch_patches = F.unfold(
        sketch_rgb.float(),
        (cell_height, cell_width),
        stride=(cell_height, cell_width),
    )
    num_cells = grid_side * grid_side
    modality_mask = torch.randint(
        0,
        2,
        (batch_size, num_cells),
        generator=generator,
        device=photo.device,
        dtype=torch.bool,
    )
    patches = torch.where(
        modality_mask.unsqueeze(1),
        sketch_patches,
        photo_patches,
    )  # [B, 3*cell_height*cell_width, num_cells]

    puzzle = F.fold(
        patches,
        output_size=(height, width),
        kernel_size=(cell_height, cell_width),
        stride=(cell_height, cell_width),
    )
    if not batched:
        puzzle = puzzle.squeeze(0)
        modality_mask = modality_mask.squeeze(0)
    return puzzle, modality_mask
