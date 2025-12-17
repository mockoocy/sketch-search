from sqlmodel import Session, select

from server.auth.otp.models import OtpCode


class SqlOtpRepository:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def find_otp_by_challenge_hash(self, challenge_token: str) -> OtpCode | None:
        statement = select(OtpCode).where(
            OtpCode.challenge_token_hash == challenge_token,
        )
        results = self.db_session.exec(statement)
        return results.first()

    def create_otp(self, code: OtpCode) -> OtpCode:
        self.db_session.add(code)
        self.db_session.commit()
        self.db_session.refresh(code)
        return code

    def update_otp(self, code: OtpCode) -> OtpCode:
        self.db_session.add(code)
        self.db_session.commit()
        self.db_session.refresh(code)
        return code
