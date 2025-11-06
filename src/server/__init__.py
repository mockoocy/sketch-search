import uvicorn

from server.config_model import get_server_config
from server.server import APP


def serve() -> None:
    uvicorn.run(
        APP,
        host=get_server_config().host,
        port=get_server_config().port,
        log_level=get_server_config().log_level,
        timeout_graceful_shutdown=5,
    )
