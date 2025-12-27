from io import BytesIO
from pathlib import Path

import numpy as np
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
from server.index.utils import create_content_hash
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

    def add_image(self, image: bytes, relative_path: Path) -> None:
        with BytesIO(image) as buf:
            img = Image.open(buf)
            try:
                img.verify()
            except Exception as ex:
                err_msg = f"Image verification failed for file: {relative_path}"
                raise InvalidImageError(
                    err_msg,
                ) from ex

        self._image_repository.write_image(image, relative_path)
        app_logger.info("Added image at path: %s", relative_path)

    def add_thumbnail_for_image(self, relative_path: Path) -> None:
        try:
            image_bytes = self._image_repository.read_image(relative_path)
        except InvalidFsAccessError as ex:
            err_msg = f"Image does not exist: {relative_path}"
            raise ImageNotFoundError(err_msg) from ex

        with BytesIO(image_bytes) as buf:
            img = Image.open(buf)
            try:
                img.verify()
            except Exception as ex:
                err_msg = f"Image verification failed for file: {relative_path}"
                raise InvalidImageError(
                    err_msg,
                ) from ex

        with BytesIO(image_bytes) as buf:
            thumb = Image.open(buf)
            thumb.thumbnail(self._thumbnail_config.size)
            out = BytesIO()
            thumb.save(out, format=img.format)
            thumb_bytes = out.getvalue()

        self._image_repository.write_thumbnail(thumb_bytes, relative_path)

    def remove_thumbnail_for_image(self, relative_path: Path) -> None:
        self._image_repository.delete_thumbnail(relative_path)

    def remove_image(self, image_id: int) -> None:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if not indexed_image:
            return

        rel = Path(indexed_image.path)

        self._indexed_image_repository.delete_image_by_path(rel)
        self._image_repository.delete_image(rel)
        self._image_repository.delete_thumbnail(rel)

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        return self._indexed_image_repository.query_images(query)

    def similarity_search(
        self,
        image: Image.Image,
        top_k: int,
        query: ImageSearchQuery,
    ) -> list[IndexedImage]:
        image_numpy = np.array(image.convert("RGB").resize((224, 224)))
        embedding = self._embedder.embed(image_numpy)[0]
        return self._indexed_image_repository.get_k_nearest_images(
            embedding,
            top_k,
            query,
        )

    def search_by_image(
        self,
        image_id: int,
        top_k: int,
        query: ImageSearchQuery,
    ) -> list[IndexedImage]:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if not indexed_image:
            err_msg = f"Indexed image with ID {image_id} not found."
            raise ImageNotFoundError(err_msg)
        return self._indexed_image_repository.get_k_nearest_images(
            indexed_image.embedding,
            top_k,
            query,
        )

    def get_unindexed_images(self) -> list[Path]:
        recursive = self._watcher_config.watch_recursive
        all_image_paths = self._image_repository.list_images(
            Path(),
            recursive=recursive,
        )

        indexed_paths = set[Path]()
        page = 1
        while True:
            batch = self._indexed_image_repository.query_images(
                ImageSearchQuery(
                    page=page,
                    items_per_page=QUERY_IMAGE_BATCH_SIZE,
                ),
            )
            indexed_paths.update(Path(img.path) for img in batch)
            if len(batch) < QUERY_IMAGE_BATCH_SIZE:
                break
            page += 1

        unindexed = [path for path in all_image_paths if path not in indexed_paths]
        app_logger.info("Found %d unindexed images.", len(unindexed))
        return unindexed

    def get_thumbnail_for_image(self, image_id: int) -> Image.Image:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if not indexed_image:
            err_msg = f"Indexed image with ID {image_id} not found."
            raise ImageNotFoundError(err_msg)

        rel = Path(indexed_image.path)

        try:
            thumb_bytes = self._image_repository.read_thumbnail(rel)
        except InvalidFsAccessError:
            self.add_thumbnail_for_image(rel)
            thumb_bytes = self._image_repository.read_thumbnail(rel)

        with BytesIO(thumb_bytes) as buf:
            img = Image.open(buf)
            img.load()
            return img

    def clean_stale_indexed_images(self) -> None:
        collection_size = self._indexed_image_repository.get_total_images_count()
        pages = (collection_size + QUERY_IMAGE_BATCH_SIZE - 1) // QUERY_IMAGE_BATCH_SIZE

        for page in range(1, pages + 1):
            batch = self._indexed_image_repository.query_images(
                ImageSearchQuery(
                    page=page,
                    items_per_page=QUERY_IMAGE_BATCH_SIZE,
                ),
            )
            for indexed_image in batch:
                rel = Path(indexed_image.path)
                try:
                    image_bytes = self._image_repository.read_image(rel)
                except InvalidFsAccessError:
                    self._indexed_image_repository.delete_image_by_path(rel)
                    app_logger.info(
                        "Removed stale indexed image due to missing file: %s",
                        rel,
                    )
                    continue

                with BytesIO(image_bytes) as buf:
                    img = Image.open(buf).convert("RGB")
                    content_hash = create_content_hash(img)

                if content_hash != indexed_image.content_hash:
                    self._indexed_image_repository.delete_image_by_path(rel)
                    app_logger.info(
                        "Removed stale indexed image due to hash mismatch: %s",
                        rel,
                    )

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        return self._indexed_image_repository.get_image_by_path(image_path)

    def get_image_by_id(self, image_id: int) -> IndexedImage | None:
        return self._indexed_image_repository.get_image_by_id(image_id)
