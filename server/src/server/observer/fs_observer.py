from datetime import UTC, datetime
from pathlib import Path

from watchdog.events import (
    DirCreatedEvent as WdDirCreatedEvent,
)
from watchdog.events import (
    DirDeletedEvent as WdDirDeletedEvent,
)
from watchdog.events import (
    DirMovedEvent as WdDirMovedEvent,
)
from watchdog.events import (
    FileCreatedEvent as WdFileCreatedEvent,
)
from watchdog.events import (
    FileDeletedEvent as WdFileDeletedEvent,
)
from watchdog.events import (
    FileMovedEvent as WdFileMovedEvent,
)
from watchdog.events import (
    FileSystemEvent as WdFileSystemEvent,
)
from watchdog.events import (
    PatternMatchingEventHandler,
)

from server.events.event_bus import EventBus
from server.events.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from server.observer.path_resolver import PathResolver

IMG_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.webp"]


class ImageWatcherHandler(PatternMatchingEventHandler):
    def __init__(self, event_bus: EventBus, path_resolver: PathResolver) -> None:
        super().__init__(
            patterns=IMG_PATTERNS,
            ignore_patterns=None,
            ignore_directories=True,
            case_sensitive=False,
        )
        self._event_bus = event_bus
        self._path_resolver = path_resolver

    def on_created(self, event: WdDirCreatedEvent | WdFileCreatedEvent) -> None:
        src_file = self._path_resolver.to_relative(Path(str(event.src_path)))
        # avoiding a race condition by using current time instead of file's mtime
        # should be very close to actual creation time, but there may be a case
        # that the file stats are not yet uploaded when file is being initially created
        created_time = datetime.now(tz=UTC)
        new_event = FileCreatedEvent(path=src_file, created_at=created_time)
        self._event_bus.publish(new_event)

    def on_deleted(self, event: WdDirDeletedEvent | WdFileDeletedEvent) -> None:
        src_file = self._path_resolver.to_relative(Path(str(event.src_path)))
        new_event = FileDeletedEvent(path=src_file)
        self._event_bus.publish(new_event)

    def on_moved(self, event: WdDirMovedEvent | WdFileMovedEvent) -> None:
        old_path = self._path_resolver.to_relative(Path(str(event.src_path)))
        new_path = self._path_resolver.to_relative(Path(str(event.dest_path)))
        moved_at = datetime.fromtimestamp(new_path.stat().st_mtime, tz=UTC)
        new_event = FileMovedEvent(
            old_path=old_path,
            new_path=new_path,
            moved_at=moved_at,
        )
        self._event_bus.publish(new_event)

    def on_modified(self, event: WdFileSystemEvent) -> None:
        src_file = self._path_resolver.to_relative(Path(str(event.src_path)))
        modified_at = datetime.fromtimestamp(src_file.stat().st_mtime, tz=UTC)
        new_event = FileModifiedEvent(path=src_file, modified_at=modified_at)
        self._event_bus.publish(new_event)
