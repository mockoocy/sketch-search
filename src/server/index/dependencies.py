from typing import Annotated

from fastapi import Depends

from server.config.yaml_loader import ServerConfigDep
from server.db_core import DbSessionDep
from server.index.embedder import Embedder
from server.index.impl.default_service import DefaultIndexingService
from server.index.impl.pgvector_repository import PgVectorIndexedImageRepository
from server.index.registry import EmbedderRegistry
from server.index.repository import IndexedImageRepository
from server.index.service import IndexingService


# this one may be a bit redundant, but let's keep it the same
# way as elsewhere in the codebase.
def _get_embedder_registry(
    server_config: ServerConfigDep,
) -> EmbedderRegistry:
    return EmbedderRegistry(server_config.embedder_registry)


EmbeddersRegistryDep = Annotated[EmbedderRegistry, Depends(_get_embedder_registry)]


def _get_embedder_dependency(registry: EmbeddersRegistryDep) -> Embedder:
    return registry.chosen_embedder


EmbedderDep = Annotated[Embedder, Depends(_get_embedder_dependency)]


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
