from typing import TYPE_CHECKING, cast

import uvicorn

from server.server import create_app

if TYPE_CHECKING:
    from server.config.models import ServerConfig


app = create_app()


def serve() -> None:
    config = cast("ServerConfig", app.state.config)
    # I want to use the same log level for uvicorn as for the app
    # at least for now...
    # Uvicorn wants lowercase log levels
    # while logging module uses uppercase :)
    log_level = config.log_level.lower()
    if config.dev:
        uvicorn.run(
            "server:app",
            host=config.host,
            port=config.port,
            log_level=log_level,
            timeout_graceful_shutdown=5,
            reload=True,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
    else:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=log_level,
            timeout_graceful_shutdown=5,
        )
