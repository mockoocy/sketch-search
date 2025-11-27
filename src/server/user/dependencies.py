from typing import Annotated

from fastapi import Depends

from server.db_core import DbSessionDep
from server.user.impl.sql_repository import SqlUserRepository
from server.user.repository import UserRepository


def _get_user_repository(session: DbSessionDep) -> UserRepository:
    return SqlUserRepository(session)


UserRepositoryDep = Annotated[UserRepository, Depends(_get_user_repository)]
