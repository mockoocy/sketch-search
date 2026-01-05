import asyncio
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.config.models import NoAuthConfig, ServerConfig, WatcherConfig
from server.server import create_app

if TYPE_CHECKING:
    from server.index.service import IndexingService


class DummyIndexingService:
    def __init__(self) -> None:
        self.call_count = 0
        self.call_args = list[list[Path]]()

    def embed_images(self, image_paths: list[Path]) -> None:
        self.call_count += 1
        self.call_args.append(image_paths)
        sleep(0.25)  # Simulate some processing time

    def remove_image(self, image_path: Path) -> None: ...

    def move_image(self, old_path: Path, new_path: Path) -> None: ...


@pytest.mark.asyncio
async def test_reacts_to_events(settings: ServerConfig, tmp_path: Path) -> None:
    new_settings = settings.copy(
        update={
            "auth": NoAuthConfig(),
            "watcher": WatcherConfig(
                files_batch_size=2,
                watched_directory=tmp_path.as_posix(),
            ),
        },
    )
    app = create_app(new_settings)
    with TestClient(app) as _client:
        img_1 = Image.new("RGB", (10, 10), color="red")
        img_1_path = tmp_path / "img_1.jpg"
        img_1.save(img_1_path)
        img_2 = Image.new("RGB", (10, 10), color="blue")
        img_2_path = tmp_path / "img_2.jpg"
        img_2.save(img_2_path)
        img_3 = Image.new("RGB", (10, 10), color="green")
        img_3_path = tmp_path / "img_3.jpg"
        img_3.save(img_3_path)
        indexing_service: IndexingService = app.state.indexing_service
        start_size = indexing_service.get_collection_size()
        await asyncio.sleep(2.0)
        assert indexing_service.get_collection_size() == start_size + 3
        img_1_ref = tmp_path / "img_1_ref.jpg"
        img_1_path.rename(img_1_ref)
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 3
        img_1_ref.unlink()
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 2

        img_2 = Image.new("RGB", (10, 10), color="purple")
        img_2.save(img_2_path)
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 2
