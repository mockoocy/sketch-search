from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlmodel import Session

from server.auth.otp.impl.default_service import DefaultOtpAuthService
from server.auth.otp.impl.smtp_sender import SmtpOtpSender
from server.auth.otp.impl.sql_repository import SqlOtpRepository
from server.auth.otp.routes import otp_router
from server.auth.otp.sender import OtpSender
from server.config.models import ServerConfig
from server.db_core import get_db_engine, init_db
from server.events.event_bus import EventBus
from server.images.impl.default_service import DefaultImageService
from server.images.impl.fs_repository import FsImageRepository
from server.images.routes import images_router
from server.images.service import ImageService
from server.index.embedder import Embedder
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.registry import EmbedderRegistry
from server.index.service import IndexingService
from server.logger import app_logger
from server.observer.background_embedder import BackgroundEmbedder
from server.observer.path_resolver import PathResolver
from server.observer.routes import observer_router
from server.routes import health_router
from server.session.impl.default_service import DefaultSessionService
from server.session.impl.sql_repository import SqlSessionRepository
from server.session.routes import session_router
from server.user.impl.default_service import DefaultUserService
from server.user.impl.sql_repository import SqlUserRepository
from server.user.routes import user_router


async def bootstrap_index(
    image_service: ImageService,
    background_embedder: BackgroundEmbedder,
    embedder: Embedder,
    indexing_service: IndexingService,
) -> None:
    app_logger.info("Bootstrapping image index...")
    image_service.clean_stale_indexed_images()
    missing_images = image_service.get_unindexed_images()
    for path in missing_images:
        background_embedder.enqueue_file(path)
        image_service.add_thumbnail_for_image(path)
    to_reindex = indexing_service.paths_to_reindex(embedder.name)
    for path in to_reindex:
        background_embedder.enqueue_file(path)
    app_logger.info("Image index bootstrapping complete.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.event_bus = EventBus()
    app.state.path_resolver = PathResolver(
        watcher_config=app.state.config.watcher,
        thumbnail_config=app.state.config.thumbnail,
    )
    engine = get_db_engine(app.state.config.database)
    app.state.db_session = Session(engine)
    app_logger.setLevel(app.state.config.log_level)
    app.state.user_repository = SqlUserRepository(app.state.db_session)
    app.state.user_service = DefaultUserService(app.state.user_repository)
    app.state.session_repository = SqlSessionRepository(app.state.db_session)
    app.state.session_service = DefaultSessionService(app.state.session_repository)
    app.state.indexed_image_repository = PgVectorIndexedImageRepository(
        app.state.db_session,
        watched_directory=Path(app.state.config.watcher.watched_directory),
    )

    app.state.embedder_registry = EmbedderRegistry(app.state.config.embedder_registry)
    app.state.embedder = app.state.embedder_registry.chosen_embedder
    if app.state.config.auth.kind == "otp":
        app.state.otp_auth_repository = SqlOtpRepository(app.state.db_session)
        app.state.otp_auth_service = DefaultOtpAuthService(
            session_config=app.state.config.session,
            otp_repository=app.state.otp_auth_repository,
            user_repository=app.state.user_repository,
            otp_sender=app.state.otp_sender,
        )

    app.state.image_repository = FsImageRepository(
        path_resolver=app.state.path_resolver,
    )
    app.state.indexing_service = DefaultIndexingService(
        indexed_repository=app.state.indexed_image_repository,
        embedder=app.state.embedder,
        path_resolver=app.state.path_resolver,
        image_repository=app.state.image_repository,
    )
    app.state.image_service = DefaultImageService(
        image_repository=app.state.image_repository,
        indexed_image_repository=app.state.indexed_image_repository,
        embedder=app.state.embedder,
        thumbnail_config=app.state.config.thumbnail,
        watcher_config=app.state.config.watcher,
    )
    app.state.background_embedder = BackgroundEmbedder(
        config=app.state.config.watcher,
        indexing_service=app.state.indexing_service,
        image_service=app.state.image_service,
        event_bus=app.state.event_bus,
        path_resolver=app.state.path_resolver,
    )
    app.state.background_embedder.start()
    init_db(app.state.config)
    await bootstrap_index(
        image_service=app.state.image_service,
        background_embedder=app.state.background_embedder,
        embedder=app.state.embedder,
        indexing_service=app.state.indexing_service,
    )
    app_logger.info("Server startup complete.")
    yield
    app.state.background_embedder.stop()
    app.state.db_session.close()
    engine.dispose()


def create_app(config: ServerConfig, *, otp_sender: OtpSender | None = None) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.config = config
    if config.auth.kind == "otp":
        app.state.otp_sender = otp_sender or SmtpOtpSender(
            config=app.state.config.auth.smtp,
        )
    app.include_router(health_router)
    if app.state.config.auth.kind == "otp":
        app.include_router(otp_router)
        app.include_router(user_router)
    app.include_router(session_router)
    app.include_router(images_router)
    app.include_router(observer_router)

    # Mapping HTTPException to JSON responses
    # to stay consistent with the "error" field.
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,  # noqa: ARG001
        exc: HTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
            },
        )

    return app
