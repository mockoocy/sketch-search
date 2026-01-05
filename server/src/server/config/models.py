import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from pydantic_settings.sources import YamlConfigSettingsSource

# Based on the ones supported by the uvicorn logger
type LogLevel = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
]

logging.getLogger("sketch-search").setLevel(logging.INFO)


class SmtpConfig(BaseModel):
    host: str
    port: int = Field(default=587, ge=1, le=65535)
    use_tls: bool = Field(default=True)
    username: str
    password: str
    from_address: EmailStr


class OtpAuthConfig(BaseModel):
    smtp: SmtpConfig
    default_user_email: EmailStr
    kind: Literal["otp"] = Field(default="otp")
    # Number of allowed attempts before invalidating the OTP
    expires_in_s: int = Field(default=300)
    max_attempts: int = Field(default=5)


class NoAuthConfig(BaseModel):
    kind: Literal["none"] = Field(default="none")


type AuthConfig = OtpAuthConfig | NoAuthConfig


class SessionConfig(BaseModel):
    expires_in_s: int = Field(default=3600)  # 1 hour
    secret_key: str = Field(default="your-session-secret-key")


class EmbedderConfigFile(BaseModel):
    file: Path  # path to a python module
    class_name: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None


class EmbedderConfigDotted(BaseModel):
    target: str  # module.submodule.ClassName format
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None


type EmbedderConfig = EmbedderConfigFile | EmbedderConfigDotted


class EmbedderRegistryConfig(BaseModel):
    embedders: dict[str, EmbedderConfig] = Field(default_factory=dict)
    chosen_embedder: str


class PostgresConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    # These defaults are not reasonable, but at least it makes it
    # easier to type check :)
    database: str = Field(default="postgres")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")


class WatcherConfig(BaseModel):
    watched_directory: str = Field(default="./")
    watch_recursive: bool = Field(default=False)
    files_batch_size: int = Field(default=8, ge=1)


class ThumbnailConfig(BaseModel):
    size: tuple[int, int] = Field(default=(128, 128))
    thumbnail_directory: str = Field(default="./thumbnails")


class ServerConfig(BaseSettings):
    host: str = Field(default="127.0.0.1")
    # ge 1024, because occupying well-known ports is cringe
    port: int = Field(default=8000, ge=1024, le=65535)
    log_level: LogLevel = Field(default="INFO")
    dev: bool = Field(default=False)
    auth: AuthConfig = Field(default_factory=lambda: NoAuthConfig())
    session: SessionConfig = Field(default_factory=lambda: SessionConfig())
    watcher: WatcherConfig = Field(
        default_factory=lambda: WatcherConfig(),
    )
    thumbnail: ThumbnailConfig = Field(
        default_factory=lambda: ThumbnailConfig(),
    )
    embedder_registry: EmbedderRegistryConfig = Field(
        default_factory=lambda: EmbedderRegistryConfig(
            embedders={
                "default": EmbedderConfigDotted(
                    target="server.embedders.default.DefaultEmbedder",
                ),
            },
            chosen_embedder="default",
        ),
    )
    database: PostgresConfig = PostgresConfig()

    # pydantic config - makes it immutable
    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="SERVER__",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = [
            init_settings,
        ]
        cfg_path = os.environ.get("SERVER_CONFIG_PATH")
        if cfg_path:
            path = Path(cfg_path)
            if not path.exists():
                err_msg = f"Config file not found: {path}"
                raise FileNotFoundError(err_msg)
            sources.append(YamlConfigSettingsSource(settings_cls, path))
        sources.append(env_settings)
        return tuple(sources)
