from pathlib import Path
from typing import Annotated

import yaml
from fastapi import Depends

from server.config.models import ServerConfig


def _load_config(file_path: Path) -> ServerConfig:
    if not file_path.exists() or not file_path.is_file():
        return ServerConfig()

    with Path.open(file_path) as file:
        config_data = yaml.safe_load(file) or {}
    return ServerConfig.model_validate(config_data)


_CFG_PATH = Path(__file__).parent.parent.parent / "server_config.yaml"

_SERVER_CONFIG = _load_config(_CFG_PATH)


def get_server_config() -> ServerConfig:
    return _SERVER_CONFIG


ServerConfigDep = Annotated[ServerConfig, Depends(get_server_config)]
