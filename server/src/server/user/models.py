from enum import StrEnum

from sqlmodel import Field, SQLModel


class UserRole(StrEnum):
    USER = "user"
    EDITOR = "editor"
    ADMIN = "admin"


class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    email: str
    role: UserRole = Field(default=UserRole.USER)
