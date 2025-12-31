from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from server.dependencies import user_service
from server.user.models import User, UserSearchQuery


class ListUsersResponse(BaseModel):
    users: list[User]
    total: int


user_router = APIRouter(
    prefix="/api/users",
    tags=["users"],
)


@user_router.get("/")
async def list_users(
    user_service: user_service,
    query: Annotated[UserSearchQuery, Depends()],
) -> ListUsersResponse:
    users = user_service.list_users(query)
    total = user_service.get_user_count()
    return ListUsersResponse(users=users, total=total)


@user_router.get("/{user_id}")
async def get_user(
    user_id: int,
    user_service: user_service,
) -> User:
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@user_router.post("/")
async def create_user(
    user: Annotated[User, Body()],
    user_service: user_service,
) -> User:
    return user_service.create_user(user)


@user_router.put("/{user_id}")
async def update_user(
    user_id: int,
    new_user: User,
    user_service: user_service,
) -> User:
    updated_user = user_service.update_user(user_id, new_user)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user


@user_router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    user_service: user_service,
) -> None:
    user_service.delete_user(user_id)
