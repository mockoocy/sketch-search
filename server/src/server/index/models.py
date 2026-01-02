from datetime import UTC, datetime
from typing import Annotated, Any, cast
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt
from pgvector.sqlalchemy import Vector
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
from sqlmodel import Column, Field, SQLModel


def _now_factory() -> datetime:
    return datetime.now(UTC)


def _pydantic_np_array_validator(value: Any) -> npt.NDArray[np.floating]:  # noqa: ANN401
    if isinstance(value, np.ndarray):
        return cast("npt.NDArray[np.floating]", value)
    return np.array(value)


def _numpy_array_serializer(value: npt.NDArray[np.floating]) -> str:
    return str(value)


type Embedding = Annotated[
    npt.NDArray[np.floating],
    BeforeValidator(_pydantic_np_array_validator),
    PlainSerializer(_numpy_array_serializer, return_type=str),
]


class IndexedImage(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    path: str = Field(index=True, unique=True)
    embedding: Embedding = Field(sa_column=Column(Vector(1536)))
    created_at: datetime = Field(default_factory=_now_factory)
    modified_at: datetime = Field(
        default_factory=_now_factory,
        sa_column_kwargs={"onupdate": _now_factory},
    )
    user_visible_name: str
    content_hash: str
    model_name: str
    directory: str = Field(default=".")
    model_config = ConfigDict(arbitrary_types_allowed=True)
