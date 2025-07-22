import torch

from src.processing import (
    binarize_by_percentile,
    binary_thinning,
    make_jigsaw_puzzle,
)
from src.type_defs import GrayTorchBatch, RGBTorchBatch
from tests.conftest import VisualizeFunction


def test_binary_thinning(
    sketch_image_batch: GrayTorchBatch,
    visualize: VisualizeFunction | None,
) -> None:
    binary_images: GrayTorchBatch = (sketch_image_batch > 0.2).float()
    thinned_images = binary_thinning(binary_images, max_iter=1)

    assert thinned_images.shape == binary_images.shape
    assert thinned_images.dtype == torch.float32
    assert thinned_images.min() == 0.0
    assert thinned_images.max() == 1.0

    if visualize:
        thinned_as_rgb = thinned_images.unsqueeze(1).repeat(1, 3, 1, 1)
        sketches_as_rgb = sketch_image_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        visualize(sketches_as_rgb, thinned_as_rgb)


def test_binary_thinning_multiple_iterations(
    sketch_image_batch: GrayTorchBatch,
    visualize: VisualizeFunction | None,
) -> None:
    binary_images: GrayTorchBatch = (sketch_image_batch > 0.2).float()
    single_thinned_images = binary_thinning(binary_images, max_iter=1)
    thinned_images = binary_thinning(binary_images, max_iter=10)

    assert thinned_images.shape == binary_images.shape
    assert thinned_images.dtype == torch.float32
    assert thinned_images.min() == 0.0
    assert thinned_images.max() == 1.0

    assert not torch.equal(single_thinned_images, thinned_images), (
        "Thinned images should differ after multiple iterations."
    )

    assert thinned_images.mean(dim=(0, 1, 2)) < single_thinned_images.mean(
        dim=(0, 1, 2),
    ), "Amount of white pixels should decrease after multiple iterations."

    if visualize:
        thinned_as_rgb = thinned_images.unsqueeze(1).repeat(1, 3, 1, 1)
        sketches_as_rgb = sketch_image_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        visualize(sketches_as_rgb, thinned_as_rgb)


def test_percentile_binarization(
    sketch_image_batch: GrayTorchBatch,
    visualize: VisualizeFunction | None,
) -> None:
    percentile_binarized_images = binarize_by_percentile(
        sketch_image_batch,
        percentile=85.0,
    )
    assert percentile_binarized_images.shape == sketch_image_batch.shape
    assert percentile_binarized_images.dtype == torch.float32
    assert percentile_binarized_images.min() == 0.0
    assert percentile_binarized_images.max() == 1.0

    if visualize:
        percentile_as_rgb = percentile_binarized_images.unsqueeze(1).repeat(1, 3, 1, 1)
        sketches_as_rgb = sketch_image_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        visualize(sketches_as_rgb, percentile_as_rgb)


def test_jigsaw_puzzle(
    photo_image_batch: RGBTorchBatch,
    sketch_image_batch: GrayTorchBatch,
    visualize: VisualizeFunction | None,
) -> None:
    grid_side = 3
    puzzle, src_mask = make_jigsaw_puzzle(
        photo_image_batch,
        sketch_image_batch,
        generator=torch.Generator().manual_seed(42),
        grid_side=grid_side,
    )

    assert puzzle.shape == (photo_image_batch.shape[0], 3, 224, 224)
    assert src_mask.shape == (photo_image_batch.shape[0], grid_side * grid_side)

    if visualize:
        sketch_rgb = sketch_image_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        visualize(photo_image_batch, sketch_rgb, puzzle)  # Visualize the jigsaw puzzle
