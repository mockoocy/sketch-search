from typing import Annotated

from fastapi import Depends

from server.db_core import DbSessionDep
from server.embedder_registry.dependencies import EmbedderDep
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.repository import IndexedImageRepository
from server.index.service import IndexingService


def _get_indexed_image_repository(db_session: DbSessionDep) -> IndexedImageRepository:
    return PgVectorIndexedImageRepository(db_session=db_session)


IndexedImageRepositoryDep = Annotated[
    IndexedImageRepository,
    Depends(_get_indexed_image_repository),
]


def _get_indexing_service(
    indexed_image_repository: IndexedImageRepositoryDep,
    embedder: EmbedderDep,
) -> IndexingService:
    return DefaultIndexingService(
        repository=indexed_image_repository,
        embedder=embedder,
    )


IndexingServiceDep = Annotated[IndexingService, Depends(_get_indexing_service)]
