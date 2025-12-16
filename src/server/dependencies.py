"""
Dependencies for FastAPI routes.
"""

from typing import Annotated, cast

from fastapi import Depends, Request

from server.auth.otp.service import OtpAuthService
from server.config.models import ServerConfig
from server.images.service import ImageService
from server.index.service import IndexingService
from server.session.service import SessionService


def _get_otp_auth_service(request: Request) -> OtpAuthService:
    return cast("OtpAuthService", request.app.state.otp_auth_service)


otp_auth_service = Annotated[OtpAuthService, Depends(_get_otp_auth_service)]


def _get_session_service(request: Request) -> SessionService:
    return cast("SessionService", request.app.state.session_service)


session_service = Annotated[SessionService, Depends(_get_session_service)]


def _get_image_service(request: Request) -> ImageService:
    return cast("ImageService", request.app.state.image_service)


image_service = Annotated[ImageService, Depends(_get_image_service)]


def _get_indexing_service(request: Request) -> IndexingService:
    return cast("IndexingService", request.app.state.indexing_service)


indexing_service = Annotated[IndexingService, Depends(_get_indexing_service)]


def _get_server_config(request: Request) -> ServerConfig:
    return cast("ServerConfig", request.app.state.config)


server_config = Annotated[ServerConfig, Depends(_get_server_config)]
