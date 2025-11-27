import hashlib
import secrets

from server.config.models import ServerConfig
from server.session.models import SessionToken
from server.session.repository import SessionRepository
from server.user.models import User


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _generate_session_token() -> tuple[str, str]:
    raw = secrets.token_bytes(32)
    token = raw.hex()
    token_hash = _hash_session_token(token)
    return token, token_hash


class DefaultSessionService:
    def __init__(
        self,
        server_config: ServerConfig,
        session_repository: SessionRepository,
    ) -> None:
        self.server_config = server_config
        self.session_repository = session_repository

    def issue_token(self, user: User) -> str:
        """
        Issues an authentication token for an user.

        Args:
            user: User to get the token for

        Returns:
            Authentication token for the user
        """
        token, token_hash = _generate_session_token()
        session_token = SessionToken(token_hash=token_hash, user_id=user.id)
        self.session_repository.save_token(session_token)
        return token

    def validate_token(self, token: str) -> User | None:
        """
        Validates an authentication token and returns the associated user if valid.

        Args:
            token: Authentication token to validate

        Returns:
            User associated with the token if valid, otherwise None
        """
        token_hash = _hash_session_token(token)
        return self.session_repository.get_user_by_token_hash(token_hash)

    def revoke_token(self, token: str) -> None:
        """
        Revokes an authentication token.

        Args:
            token: Authentication token to revoke
        """
        token_hash = _hash_session_token(token)
        self.session_repository.delete_token(token_hash=token_hash)

    def refresh_token(self, token: str) -> str:
        """
        Refreshes an authentication token.

        Args:
            token: Authentication token to refresh

        Returns:
            New authentication token
        """
