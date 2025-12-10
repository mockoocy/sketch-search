from pydantic import EmailStr

from server.auth.otp.models import OtpCode
from server.session.models import SessionToken
from server.user.models import User


class DummyOtpSender:
    def __init__(self) -> None:
        self.otp_container = dict[str, str]()

    def send_otp(self, email: str, code: str) -> None:
        self.otp_container[email] = code


class DummySessionRepository:
    def __init__(self) -> None:
        self._sessions: list[SessionToken] = []

    def get_user_by_token_hash(self, token_hash: str) -> User | None:
        for session in self._sessions:
            if session.token_hash == token_hash:
                return User(id=session.user_id, email="dummy@example.com")
        return None

    def save_token(self, token: SessionToken) -> SessionToken:
        self._sessions.append(token)
        return token

    def get_token_by_hash(self, token_hash: str) -> SessionToken | None:
        for session in self._sessions:
            if session.token_hash == token_hash:
                return session
        return None

    def delete_token(self, token_hash: str) -> None:
        self._sessions = [
            session for session in self._sessions if session.token_hash != token_hash
        ]


class DummyUserRepository:
    def __init__(self) -> None:
        self._users: list[User] = [User(id=1, email="dummy@example.com")]

    def get_user_by_email(self, email: EmailStr) -> User | None:
        for user in self._users:
            if user.email == email:
                return user
        return None

    def get_user_by_id(self, user_id: int) -> User | None:
        for user in self._users:
            if user.id == user_id:
                return user
        return None

    def create_user(self, user: User) -> User:
        self._users.append(user)
        return user

    def edit_user(self, user: User) -> User:
        for idx, existing_user in enumerate(self._users):
            if existing_user.id == user.id:
                self._users[idx] = user
        return user

    def delete_user(self, user: User) -> None:
        self._users = [
            existing_user
            for existing_user in self._users
            if existing_user.id != user.id
        ]


class DummyOtpRepository:
    def __init__(self) -> None:
        self._otps: list[OtpCode] = []

    def find_otp_by_challenge_hash(self, challenge_token: str) -> OtpCode | None:
        for otp in self._otps:
            if otp.challenge_token_hash == challenge_token:
                return otp
        return None

    def create_otp(self, code: OtpCode) -> OtpCode:
        self._otps.append(code)
        return code

    def update_otp(self, code: OtpCode) -> OtpCode:
        for idx, existing_otp in enumerate(self._otps):
            if existing_otp.challenge_token_hash == code.challenge_token_hash:
                self._otps[idx] = code
        return code
