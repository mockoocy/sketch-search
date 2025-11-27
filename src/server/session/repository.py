from typing import Protocol

from server.session.models import SessionToken
from server.user.models import User


class SessionRepository(Protocol):
    def get_user_by_token_hash(self, token_hash: str) -> User | None:
        """
        Retrieves user by authentication token.

        Args:
            token_hash: token to search for - as a hash

        Returns:
            User associated with the token, or None if not found
        """
        ...

    def save_token(self, token: SessionToken) -> SessionToken:
        """
        Saves authentication token.

        Args:
            token: token to be saved
            user: user concerned with authentication
        """
        ...

    def delete_token(self, token_hash: str) -> None:
        """
        Deleted an authentication token.

        Args:
            token: token to be removed
        """
        ...
