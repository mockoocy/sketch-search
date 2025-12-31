from server.user.models import User, UserSearchQuery
from server.user.repository import UserRepository


class DefaultUserService:
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

    def get_user_by_email(self, email: str) -> User | None:
        return self.user_repository.get_user_by_email(email)

    def get_user_by_id(self, user_id: int) -> User | None:
        return self.user_repository.get_user_by_id(user_id)

    def create_user(self, user: User) -> User:
        return self.user_repository.create_user(user)

    def update_user(self, user_id: int, new_user: User) -> User:
        existing_user = self.user_repository.get_user_by_id(user_id)
        if not existing_user:
            err_msg = f"User with id {user_id} not found"
            raise ValueError(err_msg)

        return self.user_repository.edit_user(new_user)

    def delete_user(self, user_id: int) -> None:
        existing_user = self.user_repository.get_user_by_id(user_id)
        if not existing_user:
            err_msg = f"User with id {user_id} not found"
            raise ValueError(err_msg)

        self.user_repository.delete_user(existing_user)

    def list_users(self, query: UserSearchQuery) -> list[User]:
        return self.user_repository.list_users(query)

    def get_user_count(self) -> int:
        return self.user_repository.get_user_count()
