from enum import StrEnum
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class UserRole(StrEnum):
    USER = "user"
    EDITOR = "editor"
    ADMIN = "admin"

    def __lt__(self, other: "UserRole") -> bool:
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.EDITOR: 2,
            UserRole.ADMIN: 3,
        }
        return role_hierarchy[self] < role_hierarchy[other]


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str
    role: UserRole = Field(default=UserRole.USER)


class UserSearchQuery(SQLModel):
    page: int | None = 1
    page_size: int | None = 10
    email: str | None = None
    role: UserRole | None = None
