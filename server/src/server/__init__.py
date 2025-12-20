from typing import TYPE_CHECKING, cast

import uvicorn

from server.server import create_app

if TYPE_CHECKING:
    from server.config.models import ServerConfig


app = create_app()


def serve() -> None:
    config = cast("ServerConfig", app.state.config)
    if config.dev:
        uvicorn.run(
            "server:app",
            host=config.host,
            port=config.port,
            log_level=config.log_level,
            timeout_graceful_shutdown=5,
            reload=True,
        )
    else:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level,
            timeout_graceful_shutdown=5,
        )
