import contextlib
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from pydantic import BaseModel

from server.auth.guard import auth_guard
from server.dependencies import user_service
from server.user.models import User, UserRole, UserSearchQuery


class ListUsersResponse(BaseModel):
    users: list[User]
    total: int


user_router = APIRouter(
    prefix="/api/users",
    tags=["users"],
)


@user_router.get("", dependencies=[auth_guard(UserRole.ADMIN)])
async def list_users(
    user_service: user_service,
    query: Annotated[UserSearchQuery, Depends()],
) -> ListUsersResponse:
    users = user_service.list_users(query)
    total = user_service.get_user_count()
    return ListUsersResponse(users=users, total=total)


@user_router.get("/{user_id}", dependencies=[auth_guard(UserRole.ADMIN)])
async def get_user(
    user_id: UUID,
    user_service: user_service,
) -> User:
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@user_router.post("/", dependencies=[auth_guard(UserRole.ADMIN)])
async def create_user(
    user: Annotated[User, Body()],
    user_service: user_service,
    response: Response,
) -> User:
    response.status_code = 201
    return user_service.create_user(user)


@user_router.put("/{user_id}", dependencies=[auth_guard(UserRole.ADMIN)])
async def update_user(
    user_id: UUID,
    new_user: User,
    user_service: user_service,
    response: Response,
) -> User:
    updated_user = user_service.update_user(user_id, new_user)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    response.status_code = 201
    return updated_user


@user_router.delete("/{user_id}", dependencies=[auth_guard(UserRole.ADMIN)])
async def delete_user(
    user_id: UUID,
    user_service: user_service,
    response: Response,
) -> None:
    with contextlib.suppress(ValueError):
        user_service.delete_user(user_id)
    response.status_code = 204
