from fastapi import APIRouter, Request, Response
from pydantic import EmailStr

from server.dependencies import otp_auth_service, session_service

otp_router = APIRouter(
    prefix="/api/auth/otp",
    tags=["otp auth"],
)


@otp_router.get("/start")
async def start_otp_process(
    email: EmailStr,
    response: Response,
    otp_service: otp_auth_service,
) -> dict[str, str]:
    challenge_token = otp_service.start(email=email)
    response.set_cookie(
        key="challenge_token",
        value=challenge_token,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"message": "OTP process started. Check your email for the code."}


@otp_router.post("/verify")
async def verify_otp_code(
    code: str,
    request: Request,
    response: Response,
    otp_service: otp_auth_service,
    session_service: session_service,
) -> dict[str, str]:
    challenge_token = request.cookies.get("challenge_token")
    if not challenge_token:
        response.status_code = 403
        return {
            "error": "Challenge token is missing. Please start the OTP process again.",
        }
    user = otp_service.verify(code=code, challenge_token=challenge_token)
    session_token = session_service.issue_token(user=user)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    response.delete_cookie(key="challenge_token")
    return {"message": "OTP verified successfully. You are now logged in."}
