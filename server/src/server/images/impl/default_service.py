from io import BytesIO
from pathlib import Path

from PIL import Image

from server.config.models import ThumbnailConfig, WatcherConfig
from server.images.errors import (
    ImageNotFoundError,
    InvalidFsAccessError,
    InvalidImageError,
)
from server.images.models import ImageSearchQuery
from server.images.repository import ImageRepository
from server.index.embedder import Embedder
from server.index.models import IndexedImage
from server.index.repository import IndexedImageRepository
from server.logger import app_logger

QUERY_IMAGE_BATCH_SIZE = 1000


class DefaultImageService:
    def __init__(
        self,
        image_repository: ImageRepository,
        indexed_image_repository: IndexedImageRepository,
        embedder: Embedder,
        thumbnail_config: ThumbnailConfig,
        watcher_config: WatcherConfig,
    ) -> None:
        self._image_repository = image_repository
        self._indexed_image_repository = indexed_image_repository
        self._embedder = embedder
        self._thumbnail_config = thumbnail_config
        self._watcher_config = watcher_config

    def _write_to(self, base_dir: Path, relative_path: Path, image: bytes) -> None:
        base_dir_resolved = base_dir.resolve()
        full_path = (base_dir_resolved / relative_path).resolve()
        if not full_path.is_relative_to(base_dir_resolved):
            err_msg = f"Attempted to write outside the base directory: {full_path}"
            raise InvalidFsAccessError(err_msg)
        self._image_repository.write_file(image, full_path)

    def _resolve_thumbnail_path(self, full_image_path: Path) -> Path:
        thumbnail_directory = Path(self._thumbnail_config.thumbnail_directory).resolve()
        relative_path = full_image_path.relative_to(
            Path(self._watcher_config.watched_directory).resolve(),
        )
        return (thumbnail_directory / relative_path).resolve()

    def add_image(self, image: bytes, relative_path: Path) -> None:
        with BytesIO(image) as img_buffer:
            img = Image.open(img_buffer)
        try:
            img.verify()
        except Exception as ex:
            err_msg = f"Image verification failed for file: {relative_path}"
            raise InvalidImageError(err_msg) from ex
        watched_directory = Path(self._watcher_config.watched_directory).resolve()
        self._write_to(watched_directory, relative_path, image)

        with BytesIO(image) as thumb_buffer:
            thumbnail_img = Image.open(thumb_buffer)
            thumbnail_img.thumbnail(self._thumbnail_config.size)
            thumbnail_img.save(thumb_buffer, format=img.format)
            thumb_data = thumb_buffer.getvalue()
        thumbnail_directory = Path(self._thumbnail_config.thumbnail_directory).resolve()
        self._write_to(thumbnail_directory, relative_path, thumb_data)

    def remove_image(self, image_id: int) -> None:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if indexed_image:
            img_path = Path(indexed_image.path).resolve()
            thumb_path = self._resolve_thumbnail_path(img_path)
            self._image_repository.delete_file(img_path)
            self._image_repository.delete_file(thumb_path)
            self._indexed_image_repository.delete_image_by_path(img_path)

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        return self._indexed_image_repository.query_images(query)

    def similarity_search(self, image: bytes, top_k: int) -> list[IndexedImage]:
        embedding = self._embedder.embed([image])[0]
        return self._indexed_image_repository.get_k_nearest_images(embedding, top_k)

    def search_by_image(self, image_id: int, top_k: int) -> list[IndexedImage]:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if not indexed_image:
            err_msg = f"Indexed image with ID {image_id} not found."
            raise ImageNotFoundError(err_msg)
        embedding = indexed_image.embedding
        return self._indexed_image_repository.get_k_nearest_images(embedding, top_k)

    def get_unindexed_images(self) -> list[Path]:
        watched_directory = Path(self._watcher_config.watched_directory).resolve()
        recursive = self._watcher_config.watch_recursive
        all_image_paths = self._image_repository.list_images(
            watched_directory,
            recursive=recursive,
        )
        indexed_image_paths = set[Path]()
        while True:
            batch = self._indexed_image_repository.query_images(
                ImageSearchQuery(
                    page=len(indexed_image_paths) // QUERY_IMAGE_BATCH_SIZE + 1,
                    items_per_page=QUERY_IMAGE_BATCH_SIZE,
                ),
            )
            indexed_image_paths.update(Path(img.path).resolve() for img in batch)
            if len(batch) < QUERY_IMAGE_BATCH_SIZE:
                break
        unindexed_images = [
            path
            for path in all_image_paths
            if path.resolve() not in indexed_image_paths
        ]
        app_logger.info(f"Found {len(unindexed_images)} unindexed images.")
        return unindexed_images
