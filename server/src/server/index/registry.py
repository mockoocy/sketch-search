import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from server.config.models import EmbedderConfigDotted, EmbedderRegistryConfig
from server.index.embedder import Embedder
from server.logger import app_logger


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
        app_logger.info(f"Configuring EmbedderRegistry with config: {config}")
        self._embedder = self._create_embedder()

    @property
    def chosen_embedder(self) -> Embedder:
        """Get the chosen embedder instance based on the configuration."""
        return self._embedder

    def _create_embedder(self) -> Embedder:
        embedder_config = self._config.embedders[self._config.chosen_embedder]
        if isinstance(embedder_config, EmbedderConfigDotted):
            embedder_class = _load_class_from_dotted_path(embedder_config.target)
        else:
            embedder_class = _load_class_from_file(
                Path(embedder_config.file),
                embedder_config.class_name,
            )

        app_logger.info(
            f"Loading embedder '{self._config.chosen_embedder}' "
            f"using class '{embedder_class.__name__}'",
        )
        kwargs = embedder_config.kwargs or {}
        args = embedder_config.args or []
        embedder = embedder_class(*args, **kwargs)
        embedder.name = self._config.chosen_embedder
        return embedder
