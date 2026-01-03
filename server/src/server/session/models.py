from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from server.user.models import UserRole


class SessionToken(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    token_hash: str
    user_id: UUID = Field(foreign_key="user.id", ondelete="CASCADE")
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC) + timedelta(days=1),
    )


class AnonymousSessionResponse(BaseModel):
    state: Literal["anonymous"] = "anonymous"


class ChallengedSessionResponse(BaseModel):
    state: Literal["challenge_issued"] = "challenge_issued"


class AuthenticatedSessionResponse(BaseModel):
    state: Literal["authenticated"] = "authenticated"
    role: UserRole


type SessionResponse = (
    AnonymousSessionResponse | ChallengedSessionResponse | AuthenticatedSessionResponse
)
