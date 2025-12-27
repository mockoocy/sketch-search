from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from server.images.errors import InvalidFsAccessError
from server.images.repository import ImageRepository
from server.index.embedder import Embedder
from server.index.exceptions import IndexCollisionError
from server.index.models import IndexedImage
from server.index.repository import IndexedImageRepository
from server.index.utils import create_content_hash
from server.observer.path_resolver import PathResolver


class DefaultIndexingService:
    def __init__(
        self,
        indexed_repository: IndexedImageRepository,
        embedder: Embedder,
        image_repository: ImageRepository,
        path_resolver: PathResolver,
    ) -> None:
        self._indexed_repository = indexed_repository
        self._embedder = embedder
        self._image_repository = image_repository
        self._path_resolver = path_resolver

    def embed_images(self, relative_paths: list[Path]) -> None:
        items = list[tuple[Path, Image.Image, str]]()

        for rel in relative_paths:
            try:
                data = self._image_repository.read_image(rel)
            except InvalidFsAccessError:
                existing = self._indexed_repository.get_image_by_path(rel)
                if existing is not None:
                    self._indexed_repository.delete_image_by_path(rel)
                continue

            with BytesIO(data) as buf:
                img = Image.open(buf).convert("RGB")
                img.load()

            content_hash = create_content_hash(img)
            existing = self._indexed_repository.get_image_by_path(rel)
            if (
                existing is not None
                and existing.content_hash == content_hash
                and existing.model_name == self._embedder.name
            ):
                continue

            items.append((rel, img, content_hash))

        if not items:
            return

        images_numpy = np.array(
            [
                np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
                for _, img, _ in items
            ],
            dtype=np.float32,
        )

        embeddings = self._embedder.embed(images_numpy)

        to_add = list[IndexedImage]()

        for (rel, _img, content_hash), embedding in zip(items, embeddings):
            abs_path = self._path_resolver.image_abs(rel)
            stats = abs_path.stat()
            created_at = datetime.fromtimestamp(stats.st_ctime, tz=UTC)
            modified_at = datetime.fromtimestamp(stats.st_mtime, tz=UTC)

            existing = self._indexed_repository.get_image_by_path(rel)
            if existing is not None:
                existing.embedding = np.array(embedding, dtype=np.float32)
                existing.user_visible_name = rel.name
                existing.created_at = created_at
                existing.modified_at = modified_at
                existing.content_hash = content_hash
                existing.model_name = self._embedder.name
                self._indexed_repository.update_image(existing)
                continue

            to_add.append(
                IndexedImage(
                    path=rel.as_posix(),
                    embedding=np.array(embedding, dtype=np.float32),
                    user_visible_name=rel.name,
                    created_at=created_at,
                    modified_at=modified_at,
                    content_hash=content_hash,
                    model_name=self._embedder.name,
                ),
            )

        if to_add:
            self._indexed_repository.add_images(to_add)

    def remove_image(self, relative_path: Path) -> None:
        self._indexed_repository.delete_image_by_path(relative_path)

    def move_image(self, old_path: Path, new_path: Path) -> None:
        image = self._indexed_repository.get_image_by_path(old_path)
        if image is None:
            err_msg = f"Image at path {old_path} not found in index."
            raise IndexCollisionError(err_msg)
        if self._indexed_repository.get_image_by_path(new_path) is not None:
            err_msg = f"Image at path {new_path} already exists in index."
            raise IndexCollisionError(
                err_msg,
            )
        image.path = new_path.as_posix()
        image.user_visible_name = new_path.name
        self._indexed_repository.update_image(image)

    def get_collection_size(self) -> int:
        return self._indexed_repository.get_total_images_count()
