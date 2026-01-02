"""
Dependencies for FastAPI routes.
"""

from typing import Annotated, cast

from fastapi import Depends, Request

from server.auth.otp.service import OtpAuthService
from server.config.models import ServerConfig
from server.events.event_bus import EventBus
from server.images.service import ImageService
from server.index.service import IndexingService
from server.session.service import SessionService
from server.user.service import UserService


def _get_otp_auth_service(request: Request) -> OtpAuthService | None:
    try:
        return cast("OtpAuthService", request.app.state.otp_auth_service)
    except AttributeError:
        return None


otp_auth_service = Annotated[OtpAuthService | None, Depends(_get_otp_auth_service)]


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


def _get_event_bus(request: Request) -> EventBus:
    return cast("EventBus", request.app.state.event_bus)


event_bus = Annotated[EventBus, Depends(_get_event_bus)]


def _get_user_service(request: Request) -> UserService:
    return cast("UserService", request.app.state.user_service)


user_service = Annotated[UserService, Depends(_get_user_service)]
