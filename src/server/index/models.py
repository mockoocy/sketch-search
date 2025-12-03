from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlmodel import Column, Field, SQLModel


def now_factory() -> datetime:
    return datetime.now(UTC)


class IndexedImage(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    embedding: list[float] = Field(sa_column=Column(Vector(1536)))
    created_at: datetime = Field(default_factory=now_factory)
    modified_at: datetime = Field(
        default_factory=now_factory,
        sa_column_kwargs={"onupdate": now_factory},
    )
    content_hash: str
    model_name: str
