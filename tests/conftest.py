from collections.abc import Generator

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, text
from testcontainers.postgres import PostgresContainer

PGVECTOR_IMAGE = "pgvector/pgvector:pg18-trixie"


@pytest.fixture(scope="session")
def pg_container() -> Generator[PostgresContainer]:
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as pg:
        yield pg


@pytest.fixture(scope="session")
def engine(pg_container: PostgresContainer) -> Generator[Engine]:
    engine = create_engine(pg_container.get_connection_url())

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        SQLModel.metadata.create_all(conn)

    return engine


@pytest.fixture
def db_session(engine: Engine) -> Generator[Session]:
    with Session(engine) as session:
        yield session
        session.rollback()
