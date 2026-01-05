import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.config.models import NoAuthConfig, ServerConfig, WatcherConfig
from server.server import create_app

if TYPE_CHECKING:
    from server.images.service import ImageService
    from server.index.service import IndexingService


@pytest.mark.asyncio
async def test_reacts_to_events(settings: ServerConfig, tmp_path: Path) -> None:
    watch_dir = tmp_path / "watched"
    new_settings = settings.copy(
        update={
            "auth": NoAuthConfig(),
            "watcher": WatcherConfig(
                watched_directory=str(watch_dir),
                files_batch_size=2,
            ),
        },
    )
    app = create_app(new_settings)
    with TestClient(app) as _client:
        img_1 = Image.new("RGB", (10, 10), color="red")
        img_1_path = watch_dir / "img_1.jpg"
        img_1.save(img_1_path)
        img_2 = Image.new("RGB", (10, 10), color="blue")
        img_2_path = watch_dir / "img_2.jpg"
        img_2.save(img_2_path)
        img_3 = Image.new("RGB", (10, 10), color="green")
        img_3_path = watch_dir / "img_3.jpg"
        img_3.save(img_3_path)
        indexing_service: IndexingService = app.state.indexing_service
        start_size = indexing_service.get_collection_size()
        await asyncio.sleep(2.0)
        assert indexing_service.get_collection_size() == start_size + 3
        image_service: ImageService = app.state.image_service
        indexed_img_1 = image_service.get_image_by_path(
            img_1_path.relative_to(watch_dir),
        )
        assert indexed_img_1 is not None
        img_1_ref = watch_dir / "img_1_ref.jpg"
        img_1_path.rename(img_1_ref)
        await asyncio.sleep(1.0)
        img_by_old_path = image_service.get_image_by_path(
            img_1_path.relative_to(watch_dir),
        )
        assert img_by_old_path is None
        img_1_by_new_path = image_service.get_image_by_path(
            img_1_ref.relative_to(watch_dir),
        )
        assert img_1_by_new_path is not None
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 3
        old_content_hash = img_1_by_new_path.content_hash
        img_1_content_modified = Image.new("RGB", (10, 10), color="yellow")
        img_1_content_modified.save(img_1_ref)
        await asyncio.sleep(3.0)
        new_img_1 = image_service.get_image_by_path(img_1_ref.relative_to(watch_dir))
        assert new_img_1 is not None
        assert new_img_1.content_hash != old_content_hash

        img_1_ref.unlink()
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 2

        img_2 = Image.new("RGB", (10, 10), color="purple")
        img_2.save(img_2_path)
        await asyncio.sleep(1.0)
        assert indexing_service.get_collection_size() == start_size + 2
