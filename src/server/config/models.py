from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# Based on the ones supported by the uvicorn logger
LogLevel = Literal[
    "critical",
    "error",
    "warning",
    "info",
    "debug",
    "trace",
]


class SmtpConfig(BaseModel):
    host: str
    port: int = Field(default=587, ge=1, le=65535)
    use_tls: bool = Field(default=True)
    username: str
    password: str
    from_address: EmailStr


class OtpAuthConfig(BaseModel):
    kind: Literal["otp"] = Field(default="otp")
    expires_in_s: int = Field(default=300)
    # Number of allowed attempts before invalidating the OTP
    max_attempts: int = Field(default=5)
    smtp: SmtpConfig
    db_path: Path = Field(default=Path("./app.db"))


class NoAuthConfig(BaseModel):
    kind: Literal["none"] = Field(default="none")


type AuthConfig = OtpAuthConfig | NoAuthConfig


class SessionConfig(BaseModel):
    expires_in_s: int = Field(default=3600)  # 1 hour
    secret_key: str = Field(default="your-session-secret-key")


class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    # ge 1024, because occupying well-known ports is cringe
    port: int = Field(default=8000, ge=1024, le=65535)
    watched_directory: str = Field(default="./")
    watch_recursive: bool = Field(default=True)
    log_level: LogLevel = Field(default="info")
    auth: AuthConfig = Field(default_factory=lambda: NoAuthConfig())
    session: SessionConfig = Field(default_factory=lambda: SessionConfig())
    # pydantic config - makes it immutable
    model_config = ConfigDict(frozen=True)
