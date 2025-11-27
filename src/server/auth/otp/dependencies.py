from typing import Annotated

from fastapi import Depends

from server.auth.otp.impl.default_service import DefaultOtpAuthService
from server.auth.otp.impl.sql_repository import SqlOtpRepository
from server.auth.otp.repository import OtpRepository
from server.auth.otp.service import OtpAuthService
from server.config.yaml_loader import ServerConfigDep
from server.db_core import DbSessionDep
from server.user.dependencies import UserRepositoryDep


def _get_otp_repository(
    config: ServerConfigDep,
    db_session: DbSessionDep,
) -> OtpRepository:
    if config.auth.kind != "otp":
        err_msg = "OTP authentication is not enabled in the server configuration."
        raise RuntimeError(
            err_msg,
        )
    return SqlOtpRepository(db_session=db_session)


OtpRepositoryDep = Annotated[OtpRepository, Depends(_get_otp_repository)]


def _get_otp_service(
    config: ServerConfigDep,
    user_repository: UserRepositoryDep,
    otp_repository: OtpRepositoryDep,
) -> OtpAuthService:
    if config.auth.kind != "otp":
        err_msg = "OTP authentication is not enabled in the server configuration."
        raise RuntimeError(
            err_msg,
        )
    return DefaultOtpAuthService(
        config=config,
        user_repository=user_repository,
        otp_repository=otp_repository,
    )


OtpAuthServiceDep = Annotated[OtpAuthService, Depends(_get_otp_service)]
