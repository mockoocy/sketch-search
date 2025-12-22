import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Event:
    def as_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class FileCreatedEvent(Event):
    path: Path
    created_at: datetime

    def as_json(self) -> str:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["path"] = str(self.path)
        return json.dumps(data)


@dataclass
class FileDeletedEvent(Event):
    path: Path

    def as_json(self) -> str:
        data = asdict(self)
        data["path"] = str(self.path)
        return json.dumps(data)


@dataclass
class FileModifiedEvent(Event):
    path: Path
    modified_at: datetime

    def as_json(self) -> str:
        data = asdict(self)
        data["modified_at"] = self.modified_at.isoformat()
        data["path"] = str(self.path)
        return json.dumps(data)


@dataclass
class FileMovedEvent(Event):
    old_path: Path
    new_path: Path
    moved_at: datetime

    def as_json(self) -> str:
        data = asdict(self)
        data["moved_at"] = self.moved_at.isoformat()
        data["old_path"] = str(self.old_path)
        data["new_path"] = str(self.new_path)
        return json.dumps(data)
