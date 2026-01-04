import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from server.config.models import EmbedderConfigDotted, EmbedderRegistryConfig
from server.index.embedder import Embedder


def _load_class_from_file(file: Path, class_name: str) -> type:
    module_name = f"user_embedder_{file.stem}"
    spec = spec_from_file_location(module_name, file)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)


def _load_class_from_dotted_path(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class EmbedderRegistry:
    """Registry for embedder instances."""

    def __init__(self, config: EmbedderRegistryConfig) -> None:
        self._config = config
        self._embedders: dict[str, Embedder] = self._populate_registry()

    @property
    def chosen_embedder(self) -> Embedder:
        """Get the chosen embedder instance based on the configuration."""
        return self._embedders[self._config.chosen_embedder]

    def _populate_registry(self) -> dict[str, Embedder]:
        """Populate the registry with embedder instances based on the configuration."""
        embedders: dict[str, Embedder] = {}
        for name, embedder_config in self._config.embedders.items():
            if isinstance(embedder_config, EmbedderConfigDotted):
                embedder_class = _load_class_from_dotted_path(embedder_config.target)
            else:
                embedder_class = _load_class_from_file(
                    embedder_config.file,
                    embedder_config.class_name,
                )
            kwargs = embedder_config.kwargs or {}
            embedders[name] = embedder_class(**kwargs)
            embedders[name].name = name
        return embedders
