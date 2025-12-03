from typing import Annotated

from fastapi import Depends

from server.config.yaml_loader import ServerConfigDep
from server.embedder_registry.embedder import Embedder
from server.embedder_registry.registry import EmbedderRegistry


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
