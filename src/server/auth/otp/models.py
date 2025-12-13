from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    from server.config.models import OtpAuthConfig
from server.config.models import get_server_config


class LoginFailureReason(StrEnum):
    INVALID_CODE = "invalid_code"
    EXPIRED_CODE = "expired_code"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
    UNKNOWN = "unknown"


def _create_utc_datetime() -> datetime:
    return datetime.now(UTC)


def _create_expiration_datetime() -> datetime:
    otp_config = cast("OtpAuthConfig", get_server_config().auth)
    expire_in_seconds = otp_config.expires_in_s
    return datetime.now(UTC) + timedelta(seconds=expire_in_seconds)


class LoginAttempt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    ip: str
    failure_reason: LoginFailureReason | None = None
    ts: datetime = Field(default_factory=_create_utc_datetime)


class OtpCode(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    code_hash: str
    challenge_token_hash: str
    expires_at: datetime = Field(default_factory=lambda: _create_expiration_datetime())
    consumed: bool = False
    created_at: datetime = Field(default_factory=_create_utc_datetime)
