from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from server.auth.otp.sender import OtpSender
from server.config.models import (
    EmbedderConfigFile,
    EmbedderRegistryConfig,
    NoAuthConfig,
    OtpAuthConfig,
    PostgresConfig,
    ServerConfig,
    SmtpConfig,
    ThumbnailConfig,
    WatcherConfig,
)
from server.server import create_app

PGVECTOR_IMAGE = "pgvector/pgvector:pg18-trixie"

DEFAULT_TEST_USER = "test@example.com"


class CapturingOtpSender(OtpSender):
    def __init__(self) -> None:
        self.sent = dict[str, str]()

    def send_otp(self, email: str, code: str) -> None:
        self.sent[email] = code


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str, None, None]:
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture
def settings(postgres_url: str, tmp_path: Path) -> Generator[ServerConfig, None, None]:
    db_name = postgres_url.rsplit("/", 1)[-1]
    db_user = postgres_url.split("//")[1].split(":")[0]
    db_password = postgres_url.split(":")[2].split("@")[0]
    db_host = postgres_url.split("@")[1].split(":")[0]
    db_port = postgres_url.split(":")[-1].split("/")[0]
    watched_directory = tmp_path / "watched"
    watched_directory.mkdir(parents=True, exist_ok=True)
    thumbnails_directory = tmp_path / "thumbnails"
    thumbnails_directory.mkdir(parents=True, exist_ok=True)
    return ServerConfig(
        watcher=WatcherConfig(
            watched_directory=watched_directory.as_posix(),
        ),
        thumbnail=ThumbnailConfig(
            thumbnails_directory=thumbnails_directory.as_posix(),
        ),
        embedder_registry=EmbedderRegistryConfig(
            embedders={
                "dummy": EmbedderConfigFile(
                    file=Path(__file__).parent
                    / "server"
                    / "mock"
                    / "dummy_embedder.py",
                    class_name="DummyEmbedder",
                ),
            },
            chosen_embedder="dummy",
        ),
        database=PostgresConfig(
            host=db_host,
            port=int(db_port),
            database=db_name,
            user=db_user,
            password=db_password,
        ),
        auth=OtpAuthConfig(
            default_user_email=DEFAULT_TEST_USER,
            smtp=SmtpConfig(
                host="dummy",
                port=576,
                username="dummy",
                password="dummy",  # noqa: S106
                from_address=DEFAULT_TEST_USER,
            ),
        ),
    )


@pytest.fixture
def default_client(
    settings: ServerConfig,
) -> Generator[TestClient, None, None]:
    otp_sender = CapturingOtpSender()
    app: FastAPI = create_app(settings, otp_sender=otp_sender)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def no_auth_client(
    settings: ServerConfig,
) -> Generator[TestClient, None, None]:
    new_settings = settings.model_copy(update={"auth": NoAuthConfig()})
    app = create_app(new_settings)
    with TestClient(app) as client:
        yield client
