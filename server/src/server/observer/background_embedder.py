import asyncio
from pathlib import Path

from PIL import Image
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from server.config.models import WatcherConfig
from server.events.event_bus import EventBus
from server.events.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from server.images.service import ImageService
from server.index.service import IndexingService
from server.index.utils import create_content_hash
from server.logger import app_logger
from server.observer.fs_observer import ImageWatcherHandler
from server.observer.path_resolver import PathResolver

EMBED_MAX_WAIT_SECONDS = 1.0


class BackgroundEmbedder:
    def __init__(
        self,
        config: WatcherConfig,
        indexing_service: IndexingService,
        image_service: ImageService,
        event_bus: EventBus,
        path_resolver: PathResolver,
    ) -> None:
        self._queue = asyncio.Queue[Path]()
        self._loop = asyncio.get_event_loop()
        self._worker_task: asyncio.Task[None] | None = None
        self._config = config
        self._indexing_service = indexing_service
        self._image_service = image_service
        self._event_bus = event_bus
        self._path_resolver = path_resolver

        self._observer: BaseObserver = self._create_observer()
        self._event_bus.subscribe(
            FileCreatedEvent,
            self._on_file_created,
        )
        self._event_bus.subscribe(
            FileDeletedEvent,
            self._on_file_deleted,
        )
        self._event_bus.subscribe(
            FileModifiedEvent,
            self._on_file_modified,
        )
        self._event_bus.subscribe(
            FileMovedEvent,
            self._on_file_moved,
        )

    def _on_file_created(self, event: FileCreatedEvent) -> None:
        self.enqueue_file(event.path)

    def _on_file_deleted(self, event: FileDeletedEvent) -> None:
        self._indexing_service.remove_image(event.path)
        self._image_service.remove_thumbnail_for_image(event.path)

    def _on_file_modified(self, event: FileModifiedEvent) -> None:
        original_indexed_image = self._image_service.get_image_by_path(event.path)
        if original_indexed_image is None:
            self.enqueue_file(event.path)
            return
        original_content_hash = original_indexed_image.content_hash
        new_content_hash = create_content_hash(Image.open(event.path).convert("RGB"))
        if original_content_hash != new_content_hash:
            self._indexing_service.remove_image(event.path)
            self.enqueue_file(event.path)
            self._image_service.remove_thumbnail_for_image(event.path)

    def _on_file_moved(self, event: FileMovedEvent) -> None:
        # Could consider checking if the file was modified during the move
        # to avoid unnecessary re-embedding.
        self._indexing_service.remove_image(event.old_path)
        self._image_service.remove_thumbnail_for_image(event.old_path)
        self.enqueue_file(event.new_path)

    def _create_observer(self) -> BaseObserver:
        observer = Observer()
        observer.schedule(
            ImageWatcherHandler(
                event_bus=self._event_bus,
                path_resolver=self._path_resolver,
            ),
            path=self._config.watched_directory,
            recursive=self._config.watch_recursive,
        )
        return observer

    async def _run(self) -> None:
        while True:
            deadline = self._loop.time() + EMBED_MAX_WAIT_SECONDS
            batch = list[Path]()
            while (
                self._loop.time() < deadline
                and len(batch) < self._config.files_batch_size
            ):
                timeout = deadline - self._loop.time()
                try:
                    file_path = await asyncio.wait_for(self._queue.get(), timeout)
                except TimeoutError:
                    continue
                batch.append(file_path)
            if not batch:
                continue
            app_logger.info("Embedding and indexing %d files...", len(batch))
            try:
                await self._loop.run_in_executor(
                    None,
                    self._indexing_service.embed_images,
                    batch,
                )
            except Exception:  # noqa: BLE001
                app_logger.exception("Error during background embedding")

    def enqueue_file(self, file_path: Path) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, file_path)

    def start(self) -> None:
        self._observer.start()
        self._worker_task = self._loop.create_task(self._run())
        app_logger.info("Watching %s", self._config.watched_directory)

    def stop(self) -> None:
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
        if self._worker_task:
            self._worker_task.cancel()
        self._event_bus.unsubscribe(
            FileCreatedEvent,
            self._on_file_created,
        )
        self._event_bus.unsubscribe(
            FileDeletedEvent,
            self._on_file_deleted,
        )
        self._event_bus.unsubscribe(
            FileModifiedEvent,
            self._on_file_modified,
        )
        self._event_bus.unsubscribe(
            FileMovedEvent,
            self._on_file_moved,
        )
