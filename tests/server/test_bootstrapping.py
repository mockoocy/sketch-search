from pathlib import Path
from time import sleep
from unittest.mock import MagicMock, call

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.config.models import (
    EmbedderConfigFile,
    EmbedderRegistryConfig,
    NoAuthConfig,
    ServerConfig,
)
from server.server import bootstrap_index, create_app


@pytest.mark.asyncio
async def test_bootstrap_index_enqueues_missing_images_and_reindexes(
    tmp_path: Path,
) -> None:
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"

    image_service = MagicMock()
    image_service.get_unindexed_images.return_value = [p1, p2]

    background_embedder = MagicMock()
    indexing_service = MagicMock()

    embedder = MagicMock()
    embedder.name = "dummy-model"

    await bootstrap_index(
        image_service=image_service,
        background_embedder=background_embedder,
        embedder=embedder,
        indexing_service=indexing_service,
    )

    image_service.clean_stale_indexed_images.assert_called_once()
    image_service.get_unindexed_images.assert_called_once()

    assert background_embedder.enqueue_file.call_args_list == [call(p1), call(p2)]
    assert image_service.add_thumbnail_for_image.call_args_list == [call(p1), call(p2)]

    indexing_service.reindex_images_with_different_model.assert_called_once_with(
        "dummy-model",
    )

    calls = image_service.mock_calls
    assert calls.index(call.clean_stale_indexed_images()) < calls.index(
        call.get_unindexed_images(),
    )


@pytest.mark.asyncio
async def test_bootstrap_index_when_no_missing_images_only_reindexes() -> None:
    image_service = MagicMock()
    image_service.get_unindexed_images.return_value = []

    background_embedder = MagicMock()
    indexing_service = MagicMock()

    embedder = MagicMock()
    embedder.name = "dummy-model"

    await bootstrap_index(
        image_service=image_service,
        background_embedder=background_embedder,
        embedder=embedder,
        indexing_service=indexing_service,
    )

    image_service.clean_stale_indexed_images.assert_called_once()
    image_service.get_unindexed_images.assert_called_once()

    background_embedder.enqueue_file.assert_not_called()
    image_service.add_thumbnail_for_image.assert_not_called()

    indexing_service.reindex_images_with_different_model.assert_called_once_with(
        "dummy-model",
    )


def test_stale_index(
    settings: ServerConfig,
    tmp_path: Path,
) -> None:
    new_settings = settings.copy(update={"auth": NoAuthConfig()})
    app = create_app(new_settings)
    watch_dir = tmp_path / "watched"
    with TestClient(app) as client:
        response = client.get("/api/images")
        image_red = Image.new("RGB", (32, 32), color="red")
        image_red_path = watch_dir / "red.jpg"
        image_red.save(image_red_path)
        image_blue = Image.new("RGB", (32, 32), color="blue")
        image_blue_path = watch_dir / "blue.jpg"
        image_blue.save(image_blue_path)
        sleep(1)

    image_red_path.unlink()
    image_green = Image.new("RGB", (32, 32), color="green")
    image_green_path = watch_dir / "green.jpg"
    image_green.save(image_green_path)
    with TestClient(app) as client:
        sleep(3)
        response = client.get("/api/images")
        images = response.json()["images"]
        image_names = [img["user_visible_name"] for img in images]
        assert "blue.jpg" in image_names
        assert "red.jpg" not in image_names
        assert "green.jpg" in image_names
        assert response.status_code == 200


def test_reindex_on_model_change(
    settings: ServerConfig,
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watched"
    dummy_config_file = Path(__file__).parent / "mock" / "dummy_embedder.py"
    dummy_embedder = EmbedderConfigFile(
        file=dummy_config_file,
        class_name="DummyEmbedder",
    )
    embedder_registry = EmbedderRegistryConfig(
        embedders={
            "dummy": dummy_embedder,
            "another-dummy": dummy_embedder,
        },
        chosen_embedder="dummy",
    )
    new_settings = settings.copy(
        update={"embedder_registry": embedder_registry, "auth": NoAuthConfig()},
    )
    app = create_app(new_settings)

    with TestClient(app) as client:
        image_yellow = Image.new("RGB", (32, 32), color="yellow")
        image_yellow_path = watch_dir / "yellow.jpg"
        image_yellow.save(image_yellow_path)
        sleep(1)

    newer_settings = new_settings.copy(
        update={
            "embedder_registry": embedder_registry.copy(
                update={"chosen_embedder": "another-dummy"},
            ),
        },
    )
    app = create_app(newer_settings)
    with TestClient(app) as client:
        sleep(3)
        response = client.get("/api/images")
        images = response.json()["images"]
        model_names = {img["model_name"] for img in images}
        assert "another-dummy" in model_names
        assert "dummy" not in model_names
