import asyncio
from pathlib import Path
from time import sleep

import pytest

from server.config.models import WatcherConfig
from server.events.event_bus import EventBus
from server.observer.background_embedder import BackgroundEmbedder


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
async def test_queue_is_batched(tmp_path: Path) -> None:
    indexing_service = DummyIndexingService()

    bg_embedder = BackgroundEmbedder(
        config=WatcherConfig(
            watched_directory=tmp_path.as_posix(),
            watch_recursive=False,
            files_batch_size=2,
        ),
        indexing_service=indexing_service,
        event_bus=EventBus(),
    )

    bg_embedder.start()

    file1 = tmp_path / "image1.jpg"
    file2 = tmp_path / "image2.jpg"
    file3 = tmp_path / "image3.jpg"
    file1.touch()
    file2.touch()
    file3.touch()
    await asyncio.sleep(0.1)
    assert indexing_service.call_count == 1
    assert indexing_service.call_args[0] == [file1, file2]
    await asyncio.sleep(2.0)
    assert indexing_service.call_count == 2
    assert indexing_service.call_args[1] == [file3]


@pytest.mark.asyncio
async def test_queue_debounces(tmp_path: Path) -> None:
    indexing_service = DummyIndexingService()

    bg_embedder = BackgroundEmbedder(
        config=WatcherConfig(
            watched_directory=tmp_path.as_posix(),
            watch_recursive=False,
            files_batch_size=10,
        ),
        indexing_service=indexing_service,
        event_bus=EventBus(),
    )

    bg_embedder.start()

    file1 = tmp_path / "image1.jpg"
    file2 = tmp_path / "image2.jpg"
    file1.touch()
    await asyncio.sleep(0.25)
    assert indexing_service.call_count == 0
    file2.touch()
    await asyncio.sleep(2.0)
    assert indexing_service.call_count == 1
    assert indexing_service.call_args[0] == [file1, file2]
    file3 = tmp_path / "image3.jpg"
    file3.touch()
    await asyncio.sleep(2.0)
    assert indexing_service.call_count == 2
    assert indexing_service.call_args[1] == [file3]
