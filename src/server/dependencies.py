from typing import Annotated, cast

from fastapi import Depends, Request

from server.auth.otp.service import OtpAuthService
from server.session.service import SessionService


def _get_otp_auth_service(request: Request) -> OtpAuthService:
    return cast("OtpAuthService", request.app.state.otp_auth_service)


otp_auth_service = Annotated[OtpAuthService, Depends(_get_otp_auth_service)]


def _get_session_service(request: Request) -> SessionService:
    return cast("SessionService", request.app.state.session_service)


session_service = Annotated[SessionService, Depends(_get_session_service)]
