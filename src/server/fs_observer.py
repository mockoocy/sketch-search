import logging
from enum import StrEnum
from pathlib import Path
from typing import Self

from blinker import Signal
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

IMG_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.webp"]


class ImageFsEvent(StrEnum):
    CREATED = "image_created"
    DELETED = "image_deleted"
    MOVED = "image_moved"
    CHANGED = "image_changed"

    def __new__(cls, value: str) -> Self:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._signal = Signal(value)  # noqa: SLF001
        return obj

    @property
    def signal(self) -> Signal:
        return self._signal


class ImageWatcherHandler(PatternMatchingEventHandler):
    def __init__(self) -> None:
        super().__init__(
            patterns=IMG_PATTERNS,
            ignore_patterns=None,
            ignore_directories=True,
            case_sensitive=False,
        )

    def on_created(self, event: FileSystemEvent) -> None:
        ImageFsEvent.CREATED.signal.send(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        ImageFsEvent.DELETED.signal.send(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        ImageFsEvent.MOVED.signal.send(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        ImageFsEvent.CHANGED.signal.send(event)


# imports BaseObserver, because Observer is dynamically chosen
# based on the platform, each one inherits from BaseObserver
def start_observer() -> BaseObserver:
    observer = Observer()
    observer.schedule(ImageWatcherHandler(), path=".", recursive=False)
    observer.start()
    watch_path = Path().cwd()
    logging.getLogger("uvicorn.error").info(f"Started observer on path: {watch_path}")
    return observer


def shutdown_observer(observer: BaseObserver) -> None:
    observer.stop()
    observer.join()
    logging.getLogger("uvicorn.error").info("Observer shut down.")
