from pathlib import Path

from server.images.errors import ImageFormatError, InvalidFsAccessError
from server.logger import app_logger
from server.observer.path_resolver import PathResolver

IMG_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}


class FsImageRepository:
    def __init__(self, path_resolver: PathResolver) -> None:
        self._resolver = path_resolver

    def write_image(self, image: bytes, relative_path: Path) -> None:
        self._validate_extension(relative_path)
        full_path = self._resolver.image_abs(relative_path)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(image)
        app_logger.info("Wrote image to %s", full_path)

    def write_thumbnail(self, image: bytes, relative_path: Path) -> None:
        self._validate_extension(relative_path)
        full_path = self._resolver.thumbnail_abs(relative_path)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(image)
        app_logger.info("Wrote thumbnail to %s", full_path)

    def delete_image(self, relative_path: Path) -> None:
        full_path = self._resolver.image_abs(relative_path)
        if full_path.exists():
            full_path.unlink()

    def delete_thumbnail(self, relative_path: Path) -> None:
        full_path = self._resolver.thumbnail_abs(relative_path)
        if full_path.exists():
            full_path.unlink()

    def read_image(self, relative_path: Path) -> bytes:
        full_path = self._resolver.image_abs(relative_path)
        return self._read_file(full_path)

    def read_thumbnail(self, relative_path: Path) -> bytes:
        full_path = self._resolver.thumbnail_abs(relative_path)
        return self._read_file(full_path)

    def list_images(self, base_dir: Path, *, recursive: bool) -> list[Path]:
        root = self._resolver.image_abs(base_dir)

        if not root.exists() or not root.is_dir():
            err_msg = f"Invalid directory: {root}"
            raise InvalidFsAccessError(err_msg)

        pattern = "**/*" if recursive else "*"
        results = list[Path]()
        for ext in IMG_EXTENSIONS:
            for path in root.glob(f"{pattern}.{ext}"):
                relative_path = self._resolver.to_relative(path)
                results.append(relative_path)

        return results

    def _read_file(self, full_path: Path) -> bytes:
        if not full_path.exists() or not full_path.is_file():
            err_msg = f"File does not exist: {full_path}"
            raise InvalidFsAccessError(err_msg)
        return full_path.read_bytes()

    @staticmethod
    def _validate_extension(path: Path) -> None:
        if path.suffix.lower().lstrip(".") not in IMG_EXTENSIONS:
            err_msg = f"Unsupported image format: {path.suffix}"
            raise ImageFormatError(err_msg)
