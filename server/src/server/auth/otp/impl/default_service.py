import hashlib
import hmac
import secrets
import string
from datetime import UTC, datetime, timedelta

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
from server.auth.otp.sender import OtpSender
from server.config.models import SessionConfig
from server.user.models import User
from server.user.repository import UserRepository


def _hash_otp_code(secret_key: str, challenge_token: str, otp: str) -> str:
    return hmac.new(
        secret_key.encode(),
        (challenge_token + otp).encode(),
        hashlib.sha256,
    ).hexdigest()


def _generate_otp(secret_key: str, challenge_token: str) -> tuple[str, str]:
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
        A tuple, otp code and it's hash.
    """
    otp_length = 8
    characters = string.digits + string.ascii_letters + string.punctuation
    otp = "".join(secrets.choice(characters) for _ in range(otp_length))
    return otp, _hash_otp_code(secret_key, challenge_token, otp)


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
        session_config: SessionConfig,
        user_repository: UserRepository,
        otp_repository: OtpRepository,
        otp_sender: OtpSender,
    ) -> None:
        self._session_config = session_config
        self._user_repository = user_repository
        self._otp_repository = otp_repository
        self._otp_sender = otp_sender

    def start(self, email: EmailStr) -> str:
        user = self._user_repository.get_user_by_email(email)
        if not user:
            err_msg = "User not found"
            raise UserNotFoundError(err_msg)
        token, token_hash = _generate_challenge_token()
        otp, otp_hash = _generate_otp(self._session_config.secret_key, token)
        otp_code = OtpCode(
            code_hash=otp_hash,
            user_id=user.id,
            challenge_token_hash=token_hash,
            expires_at=datetime.now(UTC)
            + timedelta(seconds=self._session_config.expires_in_s),
        )
        self._otp_repository.create_otp(otp_code)
        self._otp_sender.send_otp(email=email, code=otp)
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
        if otp_code.expires_at.replace(tzinfo=UTC) < datetime.now(UTC):
            # datetime from db is naive, so we set UTC timezone info
            err_msg = "OTP code expired"
            raise OtpExpiredError(err_msg)
        if otp_code.code_hash != _hash_otp_code(
            self._session_config.secret_key,
            challenge_token,
            code,
        ):
            err_msg = "Invalid OTP code"
            raise OtpInvalidError(err_msg)

        otp_code.consumed = True
        self._otp_repository.update_otp(otp_code)

        user = self._user_repository.get_user_by_id(otp_code.user_id)
        if not user:
            err_msg = f"User with ID {otp_code.user_id} not found"
            raise UserNotFoundError(err_msg)

        return user

    def validate_challenge_token(self, challenge_token: str) -> bool:
        challenge_token_hash = _hash_challenge_token(challenge_token)
        otp_code = self._otp_repository.find_otp_by_challenge_hash(challenge_token_hash)
        if not otp_code:
            return False
        if otp_code.consumed:
            return False
        return otp_code.expires_at.replace(tzinfo=UTC) > datetime.now(UTC)
