from dataclasses import dataclass
from datetime import datetime


@dataclass
class FileCreatedEvent:
    path: str
    created_at: datetime


@dataclass
class FileDeletedEvent:
    path: str


@dataclass
class FileModifiedEvent:
    path: str
    modified_at: datetime


@dataclass
class FileMovedEvent:
    old_path: str
    new_path: str
    moved_at: datetime


type Event = FileCreatedEvent | FileDeletedEvent | FileModifiedEvent | FileMovedEvent
