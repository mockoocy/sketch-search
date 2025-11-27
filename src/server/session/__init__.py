from typing import Annotated

from fastapi import Depends

from server.config.yaml_loader import ServerConfigDep
from server.db_core import DbSessionDep
from server.session.impl.default_service import DefaultSessionService
from server.session.impl.sql_repository import SqlSessionRepository
from server.session.repository import SessionRepository
from server.session.service import SessionService


def _get_session_repository(db_session: DbSessionDep) -> SessionRepository:
    return SqlSessionRepository(db_session=db_session)


SessionRepositoryDep = Annotated[SessionRepository, Depends(_get_session_repository)]


def _get_session_service(
    server_config: ServerConfigDep,
    session_repository: SessionRepositoryDep,
) -> SessionService:
    return DefaultSessionService(
        server_config=server_config,
        session_repository=session_repository,
    )


SessionServiceDep = Annotated[SessionService, Depends(_get_session_service)]
