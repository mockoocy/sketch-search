from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.auth.otp.routes import otp_router
from server.config.models import ServerConfig
from server.config.yaml_loader import get_server_config
from server.db_core import init_db
from server.observer.fs_observer import (
    shutdown_observer,
    start_observer,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    init_db()
    app.state.observer = start_observer()
    yield
    shutdown_observer(app.state.observer)


def create_app(server_config: ServerConfig) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.config = get_server_config()

    if server_config.auth.kind == "otp":
        app.include_router(otp_router)

    return app
