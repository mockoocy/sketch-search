from typing import TYPE_CHECKING, cast

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, col, create_engine, func, select, text

from server.user.models import User

if TYPE_CHECKING:
    from server.config.models import OtpAuthConfig
from server.config.models import PostgresConfig, ServerConfig
from server.user.models import UserRole


def _get_db_engine(postgres_config: PostgresConfig) -> Engine:
    connect_string = (
        f"postgresql+psycopg://{postgres_config.user}:"
        f"{postgres_config.password}@"
        f"{postgres_config.host}:"
        f"{postgres_config.port}/"
        f"{postgres_config.database}"
    )
    return create_engine(connect_string)


def init_db(config: ServerConfig) -> None:
    engine = _get_db_engine(config.database)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        SQLModel.metadata.create_all(conn)

    # creates a default user, if none exists
    with Session(engine) as session:
        user_count_statement = select(func.count(col(User.id)))
        user_count = session.exec(user_count_statement).one()
        if user_count == 0:
            auth_config = cast("OtpAuthConfig", config.auth)
            default_user = User(
                email=auth_config.default_user_email,
                role=UserRole.ADMIN,
            )
            session.add(default_user)
            session.commit()


def get_db_session(config: PostgresConfig) -> Session:
    engine = _get_db_engine(config)
    return Session(engine)
