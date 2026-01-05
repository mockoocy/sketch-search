from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, EmailStr

from server.auth.otp.exceptions import (
    InvalidChallengeTokenError,
    OtpConsumedError,
    OtpExpiredError,
    OtpInvalidError,
    UserNotFoundError,
)
from server.dependencies import otp_auth_service, session_service
from server.session.models import (
    AuthenticatedSessionResponse,
    ChallengedSessionResponse,
)


class ErrorResponse(BaseModel):
    error: str


class StartOtpRequest(BaseModel):
    email: EmailStr


class VerifyOtpRequest(BaseModel):
    code: str


otp_router = APIRouter(
    prefix="/api/auth/otp",
    tags=["otp auth"],
)


@otp_router.post("/start")
async def start_otp_process(
    body: StartOtpRequest,
    response: Response,
    otp_service: otp_auth_service,
) -> ChallengedSessionResponse | ErrorResponse:
    try:
        challenge_token = otp_service.start(email=body.email)
    except UserNotFoundError:
        return ChallengedSessionResponse()  # don't reveal user existence
    response.set_cookie(
        key="challenge_token",
        value=challenge_token,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return ChallengedSessionResponse()


@otp_router.post("/verify")
async def verify_otp_code(
    body: VerifyOtpRequest,
    request: Request,
    response: Response,
    otp_service: otp_auth_service,
    session_service: session_service,
) -> AuthenticatedSessionResponse | ErrorResponse:
    challenge_token = request.cookies.get("challenge_token")
    if not challenge_token:
        response.status_code = 403
        return ErrorResponse(
            error="Challenge token is missing. Please start the OTP process again.",
        )
    try:
        user = otp_service.verify(code=body.code, challenge_token=challenge_token)
    except (InvalidChallengeTokenError, OtpConsumedError, OtpExpiredError) as ex:
        response.status_code = 403
        return ErrorResponse(error=str(ex))
    except OtpInvalidError:
        response.status_code = 401
        return ErrorResponse(error="Invalid OTP code.")

    session_token = session_service.issue_token(user=user)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    response.delete_cookie(key="challenge_token")
    return AuthenticatedSessionResponse(role=user.role)
