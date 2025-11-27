from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

if TYPE_CHECKING:
    from server.config.models import OtpAuthConfig
from server.config.yaml_loader import get_server_config


def _get_db_engine() -> Engine | None:
    config = get_server_config()
    if get_server_config().auth.kind == "none":
        return None
    db_path = cast("OtpAuthConfig", config.auth).db_path
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )


def init_db() -> None:
    engine = _get_db_engine()
    if not engine:
        return
    SQLModel.metadata.create_all(engine)


def _get_session() -> Generator[Session]:
    engine = _get_db_engine()
    if not engine:
        err_msg = "Database engine is not initialized."
        raise RuntimeError(err_msg)
    with Session(engine) as session:
        yield session


DbSessionDep = Annotated[Session, Depends(_get_session)]
