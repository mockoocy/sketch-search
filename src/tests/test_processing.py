import torch

from sktr.processing import (
    binarize_by_percentile,
    binary_thinning,
)
from sktr.type_defs import GrayTorchBatch
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
