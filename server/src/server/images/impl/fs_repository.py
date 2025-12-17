from pathlib import Path

from server.images.errors import (
    ImageFormatError,
    InvalidFsAccessError,
)

IMG_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "gif"]


class FsImageRepository:
    def write_file(self, image: bytes, full_path: Path) -> None:
        if full_path.suffix.lower()[1:] not in IMG_EXTENSIONS:
            err_msg = f"Unsupported image format: {full_path.suffix}"
            raise ImageFormatError(err_msg)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(image)

    def delete_file(self, full_path: Path) -> None:
        if full_path.exists():
            full_path.unlink()

    def list_images(self, base_dir: Path, *, recursive: bool) -> list[Path]:
        if not base_dir.exists():
            err_msg = f"Directory does not exist: {base_dir}"
            raise InvalidFsAccessError(err_msg)
        if not base_dir.is_dir():
            err_msg = f"Path is not a directory: {base_dir}"
            raise InvalidFsAccessError(err_msg)
        pattern = "**/*" if recursive else "*"
        files = list[Path]()
        for ext in IMG_EXTENSIONS:
            files.extend(base_dir.glob(f"{pattern}.{ext}"))
        return files
