import hashlib
import hmac
import secrets
import smtplib
import ssl
import string
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import cast

from pydantic import EmailStr

from server.auth.otp.exceptions import (
    InvalidChallengeTokenError,
    OtpConsumedError,
    OtpExpiredError,
    OtpInvalidError,
    UserNotFoundError,
)
from server.auth.otp.models import OtpCode
from server.auth.otp.repository import OtpRepository
from server.config.models import OtpAuthConfig, ServerConfig
from server.user.models import User
from server.user.repository import UserRepository


def _generate_otp_hash(secret_key: str, challenge_token: str) -> str:
    """
    Creates a secure hash for the OTP code, along the code itself.

    Before hashing, a random OTP code is generated.
    The OTP code is then combined with the challenge token before hashing.
    This is to ensure that the OTP is tied to a specific challenge,
    essentially ensuring that

    Args:
        secret_key: The secret key used for hashing the OTP.
        challenge_token: The challenge token associated with the OTP.

    Returns:
        The secure hash of the OTP code.
    """
    otp_length = 8
    characters = string.digits + string.ascii_letters + string.punctuation
    otp = "".join(secrets.choice(characters) for _ in range(otp_length))
    return hmac.new(
        secret_key.encode(),
        (challenge_token + otp).encode(),
        hashlib.sha256,
    ).hexdigest()


def _hash_challenge_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _generate_challenge_token() -> tuple[str, str]:
    raw = secrets.token_bytes(32)
    token = raw.hex()
    token_hash = _hash_challenge_token(token)
    return token, token_hash


class DefaultOtpAuthService:
    def __init__(
        self,
        config: ServerConfig,
        user_repository: UserRepository,
        otp_repository: OtpRepository,
    ) -> None:
        self._smtp_config = cast("OtpAuthConfig", config.auth).smtp
        self._session_config = config.session
        self._user_repository = user_repository
        self._otp_repository = otp_repository

    def start(self, email: EmailStr) -> str:
        user = self._user_repository.get_user_by_email(email)
        if not user:
            err_msg = "User not found"
            raise UserNotFoundError(err_msg)
        token, token_hash = _generate_challenge_token()
        otp_hash = _generate_otp_hash(self._session_config.secret_key, token)
        otp_code = OtpCode(
            code_hash=otp_hash,
            user_id=user.id,
            challenge_token_hash=token_hash,
        )
        self._otp_repository.create_otp(otp_code)
        self._send_otp_via_email(email, token)
        return token

    def verify(self, code: str, challenge_token: str) -> User:
        challenge_token_hash = _hash_challenge_token(challenge_token)
        otp_code = self._otp_repository.find_otp_by_challenge_hash(challenge_token_hash)
        if not otp_code:
            err_msg = "OTP code not found"
            raise InvalidChallengeTokenError(err_msg)
        if otp_code.consumed:
            err_msg = "OTP code already consumed"
            raise OtpConsumedError(err_msg)
        if otp_code.expires_at < datetime.now(UTC):
            err_msg = "OTP code expired"
            raise OtpExpiredError(err_msg)
        if otp_code.code_hash != code:
            err_msg = "Invalid OTP code"
            raise OtpInvalidError(err_msg)

        otp_code.consumed = True
        self._otp_repository.update_otp(otp_code)

        user = self._user_repository.get_user_by_email(otp_code.email)
        if not user:
            err_msg = f"User with ID {otp_code.user_id} not found"
            raise UserNotFoundError(err_msg)

        return user

    def _send_otp_via_email(self, email: EmailStr, token: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = "Your OTP Code"
        msg["From"] = self._smtp_config.from_address
        msg["To"] = email
        msg.set_content(f"Your OTP code is: {token}")
        context = ssl.create_default_context()

        with smtplib.SMTP(
            host=self._smtp_config.host,
            port=self._smtp_config.port,
        ) as session:
            if self._smtp_config.use_tls:
                session.starttls(context=context)
            if self._smtp_config.username and self._smtp_config.password:
                session.login(self._smtp_config.username, self._smtp_config.password)
            session.send_message(msg)
