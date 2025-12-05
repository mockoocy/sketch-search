from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from server.config.models import WatcherConfig
from server.events.event_bus import EventBus
from server.index.service import IndexingService
from server.observer.fs_observer import ImageWatcherHandler


class BackgroundEmbedder:
    def __init__(
        self,
        config: WatcherConfig,
        indexing_service: IndexingService,
        event_bus: EventBus,
    ) -> None:
        self._files_queue: list[Path] = []
        self._config = config
        self._observer: BaseObserver = self._create_observer()
        self._indexing_service = indexing_service
        self._event_bus = event_bus

    def _create_observer(self) -> BaseObserver:
        observer = Observer()
        observer.schedule(
            ImageWatcherHandler(event_bus=self._event_bus),
            path=self._config.watched_directory,
            recursive=self._config.watch_recursive,
        )
        return observer

    def start(self) -> None:
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
