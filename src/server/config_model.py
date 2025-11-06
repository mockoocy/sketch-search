from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Based on the ones supported by the uvicorn logger
LogLevel = Literal[
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]


class SMTPConfig(BaseModel):
    host: str
    port: int = Field(default=587, ge=1, le=65535)
    use_tls: bool = Field(default=True)
    username: str
    password: str
    from_address: str


class OTPAuthConfig(BaseModel):
    kind: Literal["otp"] = Field(default="otp")
    code_length: int = Field(default=6)
    expires_in_s: int = Field(default=300)
    # Number of allowed attempts before invalidating the OTP
    max_attempts: int = Field(default=5)
    smtp: SMTPConfig
    db_path: str = Field(default="./app.db")


class NoAuthConfig(BaseModel):
    kind: Literal["none"] = Field(default="none")


class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    # Taking up reserved ports is unbelieveably cringe
    port: int = Field(default=8000, ge=1024, le=65535)
    watched_directory: str = Field(default="./")
    watch_recursive: bool = Field(default=True)
    log_level: LogLevel = Field(default="info")
    auth: OTPAuthConfig | NoAuthConfig = Field(default_factory=lambda: NoAuthConfig())

    # pydantic config - makes it immutable
    model_config = ConfigDict(frozen=True)


def _load_config(file_path: Path) -> ServerConfig:
    if not file_path.exists() or not file_path.is_file():
        return ServerConfig()

    with Path.open(file_path) as file:
        config_data = yaml.safe_load(file) or {}
    return ServerConfig.model_validate(config_data)


_CFG_PATH = Path(__file__).parent.parent / "server_config.yaml"

_SERVER_CONFIG = _load_config(_CFG_PATH)


def get_server_config() -> ServerConfig:
    return _SERVER_CONFIG
