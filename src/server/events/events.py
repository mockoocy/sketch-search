from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class Event: ...


@dataclass
class FileCreatedEvent(Event):
    path: Path
    created_at: datetime


@dataclass
class FileDeletedEvent(Event):
    path: Path


@dataclass
class FileModifiedEvent(Event):
    path: Path
    modified_at: datetime


@dataclass
class FileMovedEvent(Event):
    old_path: Path
    new_path: Path
    moved_at: datetime
