from collections.abc import Generator

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, Table, create_engine

from server.config_model import get_server_config
from server.db.auth_models import LoginAttempt, OtpCode

ALL_TABLES: list[Table] = [
    OtpCode.__table__,
    LoginAttempt.__table__,
]


def shall_use_db() -> bool:
    return get_server_config().auth.kind == "otp"


def get_db_engine() -> Engine | None:
    config = get_server_config()
    if not shall_use_db():
        return None
    db_path = config.auth.db_path  # type: ignore - we know it's there already.
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )


def init_db() -> None:
    engine = get_db_engine()
    if not engine:
        return
    SQLModel.metadata.create_all(engine, tables=ALL_TABLES)


def get_session() -> Generator[Session, None, None]:
    engine = get_db_engine()
    if not engine:
        err_msg = "Database engine is not initialized."
        raise RuntimeError(err_msg)
    with Session(engine) as session:
        yield session
