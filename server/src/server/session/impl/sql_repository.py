from datetime import UTC, datetime

from sqlmodel import Session, select

from server.session.models import SessionToken
from server.user.models import User


class SqlSessionRepository:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def save_token(self, token: SessionToken) -> SessionToken:
        self.db_session.add(token)
        self.db_session.commit()
        self.db_session.refresh(token)
        return token

    def get_user_by_token_hash(self, token_hash: str) -> User | None:
        statement = (
            select(User)
            .join(SessionToken)
            .where(
                (SessionToken.token_hash == token_hash)
                & (SessionToken.expires_at > datetime.now(tz=UTC)),
            )
        )
        return self.db_session.exec(statement).first()

    def delete_token(self, token_hash: str) -> None:
        statement = select(SessionToken).where(SessionToken.token_hash == token_hash)
        token = self.db_session.exec(statement).first()
        if token:
            self.db_session.delete(token)
            self.db_session.commit()

    def get_token_by_hash(self, token_hash: str) -> SessionToken | None:
        statement = select(SessionToken).where(SessionToken.token_hash == token_hash)
        return self.db_session.exec(statement).first()
