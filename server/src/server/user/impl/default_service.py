from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from server.user.models import User, UserSearchQuery
from server.user.repository import UserRepository


class DefaultUserService:
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

    def get_user_by_email(self, email: str) -> User | None:
        return self.user_repository.get_user_by_email(email)

    def get_user_by_id(self, user_id: UUID) -> User | None:
        return self.user_repository.get_user_by_id(user_id)

    def create_user(self, user: User) -> User:
        try:
            return self.user_repository.create_user(user)
        except IntegrityError as ex:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists",
            ) from ex

    def update_user(self, user_id: UUID, new_user: User) -> User:
        existing_user = self.user_repository.get_user_by_id(user_id)
        if not existing_user:
            err_msg = f"User with id {user_id} not found"
            raise ValueError(err_msg)
        new_user.id = user_id
        try:
            return self.user_repository.edit_user(new_user)
        except IntegrityError as ex:
            raise HTTPException(
                status_code=409,
                detail="User with this email already exists",
            ) from ex

    def delete_user(self, user_id: UUID) -> None:
        existing_user = self.user_repository.get_user_by_id(user_id)
        if not existing_user:
            err_msg = f"User with id {user_id} not found"
            raise ValueError(err_msg)
        self.user_repository.delete_user(existing_user)

    def list_users(self, query: UserSearchQuery) -> list[User]:
        return self.user_repository.list_users(query)

    def get_user_count(self) -> int:
        return self.user_repository.get_user_count()
