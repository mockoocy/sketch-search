from pydantic import EmailStr
from sqlmodel import Session, select

from server.user.models import User


class SqlUserRepository:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def get_user_by_email(self, email: EmailStr) -> User | None:
        statement = select(User).where(User.email == email)
        return self.db_session.exec(statement).first()

    def get_user_by_id(self, user_id: int) -> User | None:
        statement = select(User).where(User.id == user_id)
        return self.db_session.exec(statement).first()

    def create_user(self, user: User) -> User:
        self.db_session.add(user)
        self.db_session.commit()
        self.db_session.refresh(user)
        return user

    def edit_user(self, user: User) -> User:
        self.db_session.add(user)
        self.db_session.commit()
        self.db_session.refresh(user)
        return user

    def delete_user(self, user: User) -> None:
        self.db_session.delete(user)
        self.db_session.commit()
