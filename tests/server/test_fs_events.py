from collections.abc import Generator
from pathlib import Path
from time import sleep

import pytest
from watchdog.observers import Observer

from server.events.event_bus import EventBus
from server.events.events import (
    Event,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)
from server.observer.fs_observer import ImageWatcherHandler


@pytest.fixture
def observer(tmp_path: Path) -> Generator[tuple[Path, list[Event]]]:
    event_bus = EventBus()
    handler = ImageWatcherHandler(event_bus=event_bus)

    observer = Observer()
    observer.schedule(handler, str(tmp_path))
    observer.start()

    events: list[Event] = []

    def on_created(event: FileCreatedEvent) -> None:
        events.append(event)

    def on_deleted(event: FileDeletedEvent) -> None:
        events.append(event)

    def on_moved(event: FileMovedEvent) -> None:
        events.append(event)

    def on_modified(event: FileModifiedEvent) -> None:
        events.append(event)

    event_bus.subscribe(FileCreatedEvent, on_created)
    event_bus.subscribe(FileDeletedEvent, on_deleted)
    event_bus.subscribe(FileMovedEvent, on_moved)
    event_bus.subscribe(FileModifiedEvent, on_modified)

    try:
        yield tmp_path, events
    finally:
        observer.stop()
        observer.join()


def test_created(observer: tuple[Path, list[Event]]) -> None:
    tmp_path, events = observer
    test_file = tmp_path / "new_image.jpg"
    test_file.touch()

    sleep(0.1)
    assert len(events) == 1
    assert isinstance(events[0], FileCreatedEvent)


def test_deleted(observer: tuple[Path, list[Event]]) -> None:
    tmp_path, events = observer
    test_file = tmp_path / "new_image.jpg"
    test_file.touch()
    sleep(0.1)

    test_file.unlink()
    sleep(0.1)

    assert len(events) == 2
    assert isinstance(events[0], FileCreatedEvent)
    assert isinstance(events[1], FileDeletedEvent)


def test_moved(observer: tuple[Path, list[Event]]) -> None:
    tmp_path, events = observer
    test_file = tmp_path / "new_image.jpg"
    test_file.touch()
    sleep(0.1)
    new_location = tmp_path / "moved_image.jpg"
    test_file.rename(new_location)
    sleep(0.1)

    assert len(events) == 2
    assert isinstance(events[0], FileCreatedEvent)
    assert isinstance(events[1], FileMovedEvent)


def test_modified(observer: tuple[Path, list[Event]]) -> None:
    tmp_path, events = observer
    test_file = tmp_path / "new_image.jpg"
    test_file.touch()

    with test_file.open("a") as f:
        f.write("modification")
    sleep(0.1)
    assert len(events) == 2
    assert isinstance(events[0], FileCreatedEvent)
    assert isinstance(events[1], FileModifiedEvent)
