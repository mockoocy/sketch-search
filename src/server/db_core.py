from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, col, create_engine, func, select

if TYPE_CHECKING:
    from server.config.models import OtpAuthConfig
from server.config.yaml_loader import get_server_config
from server.user.models import UserRole


def _get_db_engine() -> Engine | None:
    config = get_server_config()
    if get_server_config().auth.kind == "none":
        return None
    postgres_config = config.database
    connect_string = (
        f"postgresql+psycopg://{postgres_config.user}:"
        f"{postgres_config.password}@"
        f"{postgres_config.host}:"
        f"{postgres_config.port}/"
        f"{postgres_config.database}"
    )
    return create_engine(connect_string)


def init_db() -> None:
    engine = _get_db_engine()
    if not engine:
        return
    SQLModel.metadata.create_all(engine)

    # creates a default user, if none exists
    with Session(engine) as session:
        from server.user.models import User

        user_count_statement = select(func.count(col(User.id)))
        user_count = session.exec(user_count_statement).one()
        if user_count == 0:
            config = cast("OtpAuthConfig", get_server_config().auth)
            default_user = User(email=config.default_user_email, role=UserRole.ADMIN)
            session.add(default_user)
            session.commit()


def _get_session() -> Generator[Session]:
    engine = _get_db_engine()
    if not engine:
        err_msg = "Database engine is not initialized."
        raise RuntimeError(err_msg)
    with Session(engine) as session:
        yield session


DbSessionDep = Annotated[Session, Depends(_get_session)]
