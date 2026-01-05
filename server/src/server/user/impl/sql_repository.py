from uuid import UUID

from pydantic import EmailStr
from sqlmodel import Session, col, func, select

from server.user.models import User, UserSearchQuery


class SqlUserRepository:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def get_user_by_email(self, email: EmailStr) -> User | None:
        statement = select(User).where(User.email == email)
        return self.db_session.exec(statement).first()

    def get_user_by_id(self, user_id: UUID) -> User | None:
        statement = select(User).where(User.id == user_id)
        return self.db_session.exec(statement).first()

    def create_user(self, user: User) -> User:
        self.db_session.add(user)
        self.db_session.commit()
        self.db_session.refresh(user)
        return user

    def edit_user(self, user: User) -> User:
        select_statement = select(User).where(User.id == user.id)
        existing_user = self.db_session.exec(select_statement).first()
        if not existing_user:
            select_all_statement = select(User)
            all_users = self.db_session.exec(select_all_statement).all()
            err_msg = f"User with id {user.id} not found, existing users: {all_users}"
            raise ValueError(err_msg)
        for key, value in user.dict(exclude_unset=True).items():
            setattr(existing_user, key, value)
        self.db_session.commit()
        self.db_session.refresh(existing_user)
        return existing_user

    def delete_user(self, user: User) -> None:
        self.db_session.delete(user)
        self.db_session.commit()

    def list_users(self, query: UserSearchQuery) -> list[User]:
        conditions = []
        if query.email:
            conditions.append(User.email.ilike(f"%{query.email}%"))
        if query.role:
            conditions.append(User.role == query.role)
        statement = (
            select(User)
            .where(*conditions)
            .offset((query.page - 1) * query.page_size)
            .limit(query.page_size)
        )
        return self.db_session.exec(statement).all()

    def get_user_count(self) -> int:
        statement = select(func.count(col(User.id)))
        return self.db_session.exec(statement).one()
