from pathlib import Path
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import pytest
import torch
import torchvision.transforms.v2 as T  # noqa: N812 - widely recognized convention
from torch.utils.data import DataLoader
from torchvision import io
from torchvision.utils import make_grid

from src.photo_sketch_dataset import build_loader
from src.type_defs import Batch, GrayTorchBatch, RGBTorchBatch


class VisualizeFunction(Protocol):
    def __call__(self, *args: int) -> int: ...


IMG_DIR = Path(__file__).parent / "test_images/"
OUT_DIR = Path(__file__).parent / "test_outputs/"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _batch_images(
    imgs_paths: list[Path],
    mode: Literal["gray", "rgb"] = "rgb",
) -> RGBTorchBatch | GrayTorchBatch:
    images: list[RGBTorchBatch | GrayTorchBatch] = []
    for img_path in sorted(imgs_paths):
        io_mode = io.ImageReadMode.GRAY if mode == "gray" else io.ImageReadMode.RGB
        img = io.read_image(str(img_path), mode=io_mode)  # [B, C, H, W], C = 1 | 3
        img = img.float() / 255.0
        img = T.Resize((224, 224))(img)
        images.append(img)
    return torch.stack(images, dim=0)


@pytest.fixture(scope="session")
def sketch_image_batch() -> GrayTorchBatch:
    img_paths = list((IMG_DIR / "sketches/").glob("*.png"))
    return _batch_images(img_paths, mode="gray").squeeze(1)


@pytest.fixture(scope="session")
def photo_image_batch() -> RGBTorchBatch:
    img_paths = list((IMG_DIR / "photos").glob("*.jpg"))
    return _batch_images(img_paths)


@pytest.fixture
def visualize(
    request: pytest.FixtureRequest,
) -> VisualizeFunction | None:
    if not request.config.getoption("--visualize"):
        return None

    # creates a 3x2 image grid, with each image resized to 224x224
    # Then it is stored in OUT_DIR
    # use matplotlib to visualize the images
    def _save_grid(*batches: RGBTorchBatch) -> None:
        if not all(batch.shape == batches[0].shape for batch in batches):
            err_msg = "All input batches must have the same shape for visualization."
            raise ValueError(err_msg)

        n_cols = batches[0].shape[0]
        grid = make_grid(
            torch.cat(batches, dim=0),
            # some deranged person made it so that nrow is number of images per row....
            nrow=n_cols,
            padding=2,
        )
        img = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(n_cols * 2, 4))
        plt.title(f"{request.node.name} result")
        if img.shape[2] == 1:
            plt.imshow(img.squeeze(-1), cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()

        # THe name of the test is used as the filename
        # The value of request.node.name depends on the fixture scope
        # Thus this fixture is function-scoped.
        outfile = OUT_DIR / f"{request.node.name}"
        plt.savefig(outfile, dpi=150)
        plt.close()

    return _save_grid


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--visualize",
        action="store_true",
        help="If true, visualize the test outputs.",
    )


@pytest.fixture(scope="session", params=[False, True], ids=["plain", "jigsaw"])
def dataloader(request: pytest.FixtureRequest) -> DataLoader[Batch]:
    return build_loader(
        image_paths=sorted((IMG_DIR / "photos").glob("*.jpg")),
        sketch_paths=sorted((IMG_DIR / "sketches").glob("*.png")),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        use_jigsaw=request.param,
        grid_side=3,
    )
