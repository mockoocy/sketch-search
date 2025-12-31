from collections.abc import Callable
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException

from server.dependencies import session_service
from server.logger import app_logger
from server.user.models import UserRole


async def get_current_user_role(
    session_service: session_service,
    session_token: Annotated[str | None, Cookie()] = None,
) -> UserRole | None:
    app_logger.info("Getting current user role for session token: %s", session_token)
    if session_token is None:
        return None
    user = session_service.validate_token(session_token)
    if user is None:
        return None
    return user.role


def _require_min_role(min_role: UserRole) -> Callable[..., UserRole]:
    def dependency(
        current_role: Annotated[UserRole | None, Depends(get_current_user_role)],
    ) -> UserRole:
        if current_role is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
            )
        if current_role < min_role:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )
        return current_role

    return dependency


def auth_guard(min_role: UserRole) -> Callable[..., UserRole]:
    return Depends(_require_min_role(min_role))
