from typing import TYPE_CHECKING, cast

import uvicorn

from server.server import create_app

if TYPE_CHECKING:
    from server.config import ServerConfig


def serve() -> None:
    app = create_app()
    config = cast("ServerConfig", app.state.config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        timeout_graceful_shutdown=5,
    )
