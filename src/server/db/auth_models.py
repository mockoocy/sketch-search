from datetime import UTC, datetime, timedelta

from sqlmodel import Field, SQLModel


def _create_utc_datetime() -> datetime:
    return datetime.now(UTC)


class LoginAttempt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    ok: bool
    reason: str | None = None
    ip: str | None = None
    ts: datetime = Field(default_factory=_create_utc_datetime)


class OtpCode(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    code_hash: str
    salt: str
    expires_at: datetime
    attempts: int = 0
    consumed: bool = False
    created_at: datetime = Field(default_factory=_create_utc_datetime)


def utcnow() -> datetime:
    return _create_utc_datetime()


def plus(seconds: int) -> datetime:
    return _create_utc_datetime() + timedelta(seconds=seconds)
