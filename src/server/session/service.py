from typing import Protocol

from server.user.models import User


class SessionService(Protocol):
    def issue_token(self, user: User) -> str:
        """
        Issues an authentication token for an user.

        Args:
            user: User to get the token for

        Returns:
            Authentication token for the user
        """
        ...

    def validate_token(self, token: str) -> User | None:
        """
        Validates an authentication token and returns the associated user if valid.

        Args:
            token: Authentication token to validate

        Returns:
            User associated with the token if valid, otherwise None
        """
        ...

    def revoke_token(self, token: str) -> None:
        """
        Revokes an authentication token.

        Args:
            token: Authentication token to revoke
        """
        ...

    def refresh_token(self, token: str) -> str:
        """
        Refreshes an authentication token.

        Args:
            token: Authentication token to refresh

        Returns:
            New authentication token
        """
        ...
