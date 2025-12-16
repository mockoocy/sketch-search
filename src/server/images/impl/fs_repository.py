from io import BytesIO
from pathlib import Path

from PIL import Image

from server.config.models import WatcherConfig
from server.images.errors import (
    ImageFormatError,
    InvalidFsAccessError,
    InvalidImageError,
)

IMG_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif"]
RECURSIVE_GLOB = "**/*"
THIS_DIR_GLOB = "*"


class FsImageRepository:
    def __init__(self, watcher_config: WatcherConfig) -> None:
        self._watched_directory = Path(watcher_config.watched_directory)
        self._watch_recursive = watcher_config.watch_recursive

    def add_image_file(self, image: bytes, relative_path: Path) -> None:
        image_format = relative_path.suffix.lower()[1:]  # remove the dot
        if image_format not in IMG_EXTENSIONS:
            err_msg = f"Unsupported image format: {image_format}"
            raise ImageFormatError(err_msg)
        img = Image.open(BytesIO(image))
        try:
            img.verify()
        except Exception as ex:
            err_msg = f"Image verification failed for file: {relative_path}"
            raise InvalidImageError(err_msg) from ex
        full_path = self._watched_directory / relative_path
        with full_path.open("wb") as file:
            file.write(image)

    def remove_image_file(self, relative_path: Path) -> None:
        full_path = self._watched_directory / relative_path
        if full_path.exists():
            full_path.unlink()

    def get_images_from_directory(self, relative_dir: Path) -> list[Path]:
        directory_path = self._watched_directory / relative_dir

        if not directory_path.exists():
            err_msg = f"Directory does not exist: {directory_path}"
            raise InvalidFsAccessError(err_msg)
        if not directory_path.is_dir():
            err_msg = f"Path is not a directory: {directory_path}"
            raise InvalidFsAccessError(err_msg)
        if not directory_path.is_relative_to(self._watched_directory):
            err_msg = f"Directory is outside the watched directory: {directory_path}"
            raise InvalidFsAccessError(err_msg)
        glob_pattern = RECURSIVE_GLOB if self._watch_recursive else THIS_DIR_GLOB
        image_files: list[Path] = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(directory_path.glob(f"{glob_pattern}.{ext}"))
        return image_files
