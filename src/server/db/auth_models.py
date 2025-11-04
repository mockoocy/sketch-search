# auth_models.py
from datetime import datetime, timedelta
from typing import Optional
from sqlmodel import SQLModel, Field


class LoginAttempt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    ok: bool
    reason: Optional[str] = None
    ip: Optional[str] = None
    ts: datetime = Field(default_factory=datetime.utcnow)


class OtpCode(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True)
    code_hash: str
    salt: str
    expires_at: datetime
    attempts: int = 0
    consumed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


def utcnow() -> datetime:
    return datetime.utcnow()


def plus(seconds: int) -> datetime:
    return utcnow() + timedelta(seconds=seconds)
