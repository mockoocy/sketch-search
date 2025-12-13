from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.auth.otp.impl.default_service import DefaultOtpAuthService
from server.auth.otp.impl.smtp_sender import SmtpOtpSender
from server.auth.otp.impl.sql_repository import SqlOtpRepository
from server.auth.otp.routes import otp_router
from server.config.models import ServerConfig, get_server_config
from server.db_core import get_db_session, init_db
from server.events.event_bus import EventBus
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.registry import EmbedderRegistry
from server.observer.background_embedder import BackgroundEmbedder
from server.session.impl.default_service import DefaultSessionService
from server.session.impl.sql_repository import SqlSessionRepository
from server.user.impl.sql_repository import SqlUserRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.event_bus = EventBus()
    app.state.config = get_server_config()
    app.state.db_session = get_db_session()

    app.state.user_repository = SqlUserRepository(app.state.db_session)
    app.state.session_repository = SqlSessionRepository(app.state.db_session)
    app.state.session_service = DefaultSessionService(app.state.session_repository)
    app.state.indexed_image_repository = PgVectorIndexedImageRepository(
        app.state.db_session,
    )

    app.state.embedder_registry = EmbedderRegistry(app.state.config.embedder_registry)
    app.state.embedder = app.state.embedder_registry.chosen_embedder
    if app.state.config.auth.kind == "otp":
        app.state.otp_auth_repository = SqlOtpRepository(app.state.db_session)
        app.state.otp_sender = SmtpOtpSender(config=app.state.config.auth.smtp)
        app.state.otp_auth_service = DefaultOtpAuthService(
            session_config=app.state.config.session,
            otp_repository=app.state.otp_auth_repository,
            user_repository=app.state.user_repository,
            otp_sender=app.state.otp_sender,
        )
    app.state.index_service = DefaultIndexingService(
        repository=app.state.indexed_image_repository,
        embedder=app.state.embedder,
    )
    app.state.background_embedder = BackgroundEmbedder(
        config=app.state.config.watcher,
        indexing_service=app.state.index_service,
        event_bus=app.state.event_bus,
    )
    app.state.background_embedder.start()
    init_db()
    yield
    app.state.background_embedder.stop()


def create_app(server_config: ServerConfig) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.config = get_server_config()

    if server_config.auth.kind == "otp":
        app.include_router(otp_router)

    return app
