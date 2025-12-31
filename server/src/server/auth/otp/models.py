from datetime import UTC, datetime
from enum import StrEnum

from sqlmodel import Field, SQLModel


class LoginFailureReason(StrEnum):
    INVALID_CODE = "invalid_code"
    EXPIRED_CODE = "expired_code"
    MAX_ATTEMPTS_EXCEEDED = "max_attempts_exceeded"
    UNKNOWN = "unknown"


def _create_utc_datetime() -> datetime:
    return datetime.now(UTC)


class LoginAttempt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    ip: str
    failure_reason: LoginFailureReason | None = None
    ts: datetime = Field(default_factory=_create_utc_datetime)


class OtpCode(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    code_hash: str
    challenge_token_hash: str
    expires_at: datetime
    consumed: bool = False
    created_at: datetime = Field(default_factory=_create_utc_datetime)
