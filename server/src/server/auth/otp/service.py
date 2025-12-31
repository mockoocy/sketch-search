from typing import Protocol

from pydantic import EmailStr

from server.user.models import User


class OtpAuthService(Protocol):
    def start(self, email: EmailStr) -> str: ...
    def verify(self, code: str, challenge_token: str) -> User: ...
    def validate_challenge_token(self, challenge_token: str) -> bool: ...
