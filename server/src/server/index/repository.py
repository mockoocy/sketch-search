from pathlib import Path
from typing import Protocol
from uuid import UUID

from server.images.models import ImageSearchQuery
from server.index.models import Embedding, IndexedImage


class IndexedImageRepository(Protocol):
    def add_images(self, images: list[IndexedImage]) -> None:
        """Add multiple image embeddings to the repository."""
        ...

    def delete_image_by_path(self, image_path: Path) -> None:
        """Delete an image embedding from the repository by its ID."""
        ...

    def update_images(self, images: list[IndexedImage]) -> None:
        """Update existing image embeddings in the repository."""
        ...

    def get_k_nearest_images(
        self,
        embedding: Embedding,
        k: int,
        search_query: ImageSearchQuery,
    ) -> list[IndexedImage]:
        """Retrieve the k-nearest images to the given image embedding."""
        ...

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its path."""
        ...

    def get_image_by_id(self, image_id: UUID) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its ID."""
        ...

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        """Query images based on the provided search query."""
        ...

    def get_total_images_count(self) -> int:
        """Get the total number of indexed images in the repository."""
        ...

    def get_images_with_different_model(self, model_name: str) -> list[IndexedImage]:
        """Retrieve images that were indexed with a different model than specified"""
        ...
