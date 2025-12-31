from datetime import UTC, datetime, timedelta

from sqlmodel import Field, SQLModel


class SessionToken(SQLModel, table=True):
    token_hash: str
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC) + timedelta(days=1),
    )
