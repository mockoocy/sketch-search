import asyncio
from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from server.config.models import WatcherConfig
from server.events.event_bus import EventBus
from server.events.events import FileCreatedEvent
from server.index.service import IndexingService
from server.observer.fs_observer import ImageWatcherHandler

EMBED_MAX_WAIT_SECONDS = 1.0


class BackgroundEmbedder:
    def __init__(
        self,
        config: WatcherConfig,
        indexing_service: IndexingService,
        event_bus: EventBus,
    ) -> None:
        self._queue = asyncio.Queue[Path]()
        self._loop = asyncio.get_event_loop()
        self._worker_task: asyncio.Task[None] | None = None
        self._config = config
        self._indexing_service = indexing_service
        self._event_bus = event_bus

        self._observer: BaseObserver = self._create_observer()
        self._event_bus.subscribe(
            FileCreatedEvent,
            lambda event: self.enqueue_file(event.path),
        )

    def _create_observer(self) -> BaseObserver:
        observer = Observer()
        observer.schedule(
            ImageWatcherHandler(event_bus=self._event_bus),
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
            await self._loop.run_in_executor(
                None,
                self._indexing_service.embed_images,
                batch,
            )

    def enqueue_file(self, file_path: Path) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, file_path)

    def start(self) -> None:
        self._observer.start()
        self._worker_task = self._loop.create_task(self._run())

    def stop(self) -> None:
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
        if self._worker_task:
            self._worker_task.cancel()
