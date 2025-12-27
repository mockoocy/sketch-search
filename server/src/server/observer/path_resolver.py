from pathlib import Path

from server.config.models import ThumbnailConfig, WatcherConfig
from server.images.errors import InvalidFsAccessError


class PathResolver:
    def __init__(
        self,
        watcher_config: WatcherConfig,
        thumbnail_config: ThumbnailConfig,
    ) -> None:
        self.watched_root = Path(watcher_config.watched_directory).resolve()
        self.thumbnail_root = Path(thumbnail_config.thumbnail_directory).resolve()

    def image_abs(self, relative_path: Path) -> Path:
        return self._safe_join(self.watched_root, relative_path)

    def thumbnail_abs(self, relative_path: Path) -> Path:
        return self._safe_join(self.thumbnail_root, relative_path)

    def to_relative(self, abs_path: Path) -> Path:
        return abs_path.resolve().relative_to(self.watched_root)

    def _safe_join(self, base: Path, relative: Path) -> Path:
        full = (base / relative).resolve()
        if not full.is_relative_to(base):
            err_msg = f"Path {full} escapes base directory: {base}"
            raise InvalidFsAccessError(err_msg)
        return full
