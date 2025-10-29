from pathlib import Path, PosixPath
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import pytest
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision import io
from torchvision.utils import make_grid

from sktr.photo_sketch_dataset import build_loader, get_samples_from_directories
from sktr.type_defs import GrayTorchBatch, RGBTorchBatch, Sample
from sktr.vector import EvaluationStore

IMG_SIZE = (224, 224)


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
        img = transforms.Resize(IMG_SIZE)(img)
        images.append(img)
    return torch.stack(images, dim=0)


@pytest.fixture(scope="session")
def sketch_image_batch() -> GrayTorchBatch:
    img_paths = list((IMG_DIR / "sketches/").rglob("*.png"))
    return _batch_images(img_paths, mode="gray").squeeze(1)


@pytest.fixture(scope="session")
def photo_image_batch() -> RGBTorchBatch:
    img_paths = list((IMG_DIR / "photos").rglob("*.jpg"))
    return _batch_images(img_paths)


@pytest.fixture
def visualize(
    request: pytest.FixtureRequest,
) -> VisualizeFunction | None:
    if not request.config.getoption("--visualize"):
        return None

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


@pytest.fixture(scope="session")
def dataloader(request: pytest.FixtureRequest) -> DataLoader[Sample]:
    sample_paths, _, _ = get_samples_from_directories(
        images_root=IMG_DIR / "photos",
        sketches_root=IMG_DIR / "sketches",
        val_fraction=0.0,
        test_fraction=0.0,
    )

    transform = transforms.Resize(IMG_SIZE)

    return build_loader(
        samples=sample_paths,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        photo_transform=transform,
        sketch_transform=transform,
    )


@pytest.fixture(scope="function")
def eval_store(tmp_path: PosixPath) -> EvaluationStore:
    db_path = tmp_path / "milvus_lite.db"
    return EvaluationStore(embedding_size=2, store_path=db_path)
