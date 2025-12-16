import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.dates import UTC
from PIL import Image

from server.index.embedder import Embedder
from server.index.exceptions import IndexCollisionError
from server.index.models import IndexedImage
from server.index.repository import IndexedImageRepository
from server.index.utils import create_content_hash


def _user_visible_name_from_path(path: Path) -> str:
    return f"{uuid.uuid4()}_{path.name}"


class DefaultIndexingService:
    def __init__(self, repository: IndexedImageRepository, embedder: Embedder) -> None:
        self._repository = repository
        self._embedder = embedder

    def embed_images(self, relative_image_paths: list[Path]) -> None:
        """Embed a batch of images located at the given paths into the index."""

        images_bytes = [
            Image.open(path).convert("RGB").tobytes() for path in relative_image_paths
        ]
        embeddings = self._embedder.embed(images_bytes)
        images: list[IndexedImage] = []
        for path, embedding, image_bytes in zip(
            relative_image_paths,
            embeddings,
            images_bytes,
            strict=False,
        ):
            file_stats = path.stat()
            created_at = datetime.fromtimestamp(file_stats.st_ctime, tz=UTC)
            modified_at = datetime.fromtimestamp(file_stats.st_mtime, tz=UTC)
            content_hash = create_content_hash(image_bytes)
            image = IndexedImage(
                path=str(path),
                embedding=np.array(embedding, dtype=np.float32),
                user_visible_name=_user_visible_name_from_path(path),
                created_at=created_at,
                modified_at=modified_at,
                content_hash=content_hash,
                model_name=self._embedder.name,
            )
            images.append(image)
        self._repository.add_images(images)

    def remove_image(self, image_path: Path) -> None:
        """Remove an image from the index by its ID."""
        self._repository.delete_image_by_path(image_path)

    def move_image(self, old_path: Path, new_path: Path) -> None:
        image = self._repository.get_image_by_path(old_path)
        if image is None:
            err_msg = f"Image at path {old_path} not found in index."
            raise IndexCollisionError(err_msg)
        if self._repository.get_image_by_path(new_path) is not None:
            err_msg = f"Image at path {new_path} already exists in index."
            raise IndexCollisionError(err_msg)
        image.path = str(new_path)
        self._repository.update_image(image)
