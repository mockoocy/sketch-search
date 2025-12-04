from pathlib import Path

from server.config.models import (
    EmbedderConfigDotted,
    EmbedderConfigFile,
    EmbedderRegistryConfig,
)
from server.index.registry import EmbedderRegistry


def test_can_create_an_embedding() -> None:
    config = EmbedderRegistryConfig(
        embedders={
            "dummy": EmbedderConfigDotted(
                target="tests.server.dummy_embedder.DummyEmbedder",
            ),
        },
        chosen_embedder="dummy",
    )
    registry = EmbedderRegistry(config)
    embedder = registry.chosen_embedder
    # very image-like
    images = [b"image1", b"image2"]
    embeddings = embedder.embed(images)

    assert len(embeddings) == 2


def test_load_from_file() -> None:
    current_dir = Path(__file__).parent
    dummy_embedder_file = current_dir / "dummy_embedder.py"

    config = EmbedderRegistryConfig(
        embedders={
            "dummy": EmbedderConfigFile(
                file=dummy_embedder_file,
                class_name="DummyEmbedder",
            ),
        },
        chosen_embedder="dummy",
    )

    registry = EmbedderRegistry(config)
    embedder = registry.chosen_embedder
    # very image-like
    images = [b"image1", b"image2"]
    embeddings = embedder.embed(images)

    assert len(embeddings) == 2
