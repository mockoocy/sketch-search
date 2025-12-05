from pathlib import Path

from watchdog.observers import Observer

from server.events.event_bus import EventBus
from server.events.events import FileCreatedEvent
from server.observer.fs_observer import ImageWatcherHandler


def test_created(tmp_path: Path) -> None:
    event_bus = EventBus()
    handler = ImageWatcherHandler(event_bus=event_bus)
    observer = Observer()
    observer.schedule(handler, str(tmp_path))

    events: list[FileCreatedEvent] = []

    def on_file_created(event: FileCreatedEvent) -> None:
        events.append(event)

    event_bus.subscribe(FileCreatedEvent, on_file_created)

    test_file = tmp_path / "new_image.jpg"
    test_file.touch()
