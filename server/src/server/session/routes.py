from fastapi import APIRouter, Request, Response

from server.dependencies import otp_auth_service, server_config, session_service
from server.session.models import (
    AnonymousSessionResponse,
    AuthenticatedSessionResponse,
    ChallengedSessionResponse,
)
from server.user.models import UserRole

session_router = APIRouter(
    prefix="/api/session",
    tags=["session"],
)


@session_router.get("")
async def get_session_status(
    request: Request,
    response: Response,
    session_service: session_service,
    otp_service: otp_auth_service,
    server_config: server_config,
) -> (
    AnonymousSessionResponse | AuthenticatedSessionResponse | ChallengedSessionResponse
):
    # should be done with some kind of adapter.
    if server_config.auth.kind == "none":
        return AuthenticatedSessionResponse(state="authenticated", role=UserRole.ADMIN)
    if not otp_service:
        err_msg = "OTP auth service is not available? This should not happen..."
        raise RuntimeError(
            err_msg,
        )

    session_token = request.cookies.get("session_token")
    challenge_token = request.cookies.get("challenge_token")
    if not session_token and not challenge_token:
        return AnonymousSessionResponse()
    if challenge_token and otp_service.validate_challenge_token(challenge_token):
        return ChallengedSessionResponse()

    if challenge_token:
        response.delete_cookie(key="challenge_token")

    user = session_service.validate_token(session_token)
    if not user:
        response.delete_cookie(key="session_token")
        return AnonymousSessionResponse()

    return AuthenticatedSessionResponse(state="authenticated", role=user.role)


@session_router.post("/logout")
async def logout(
    response: Response,
) -> AnonymousSessionResponse:
    response.delete_cookie(key="session_token")
    response.delete_cookie(key="challenge_token")
    return AnonymousSessionResponse()
