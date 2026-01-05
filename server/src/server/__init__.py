import argparse
import os

import uvicorn

from server.config.models import ServerConfig
from server.server import create_app


def serve() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    if args.config:
        os.environ["SERVER_CONFIG_PATH"] = args.config
    config = ServerConfig()
    app = create_app(config=config)
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
