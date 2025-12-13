import uvicorn

from server.config.models import get_server_config
from server.server import create_app


def serve() -> None:
    config = get_server_config()
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        timeout_graceful_shutdown=5,
    )
