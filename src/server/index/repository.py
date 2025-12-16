from pathlib import Path
from typing import Protocol

from server.images.models import ImageSearchQuery
from server.index.models import IndexedImage


class IndexedImageRepository(Protocol):
    def add_images(self, images: list[IndexedImage]) -> None:
        """Add multiple image embeddings to the repository."""
        ...

    def delete_image_by_path(self, image_path: Path) -> None:
        """Delete an image embedding from the repository by its ID."""
        ...

    def update_image(self, image: IndexedImage) -> None:
        """Update an existing image embedding in the repository."""
        ...

    def get_k_nearest_images(
        self,
        embedding: list[float],
        k: int,
    ) -> list[IndexedImage]:
        """Retrieve the k-nearest images to the given image embedding."""
        ...

    def get_image_by_path(self, image_path: Path) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its path."""
        ...

    def get_image_by_id(self, image_id: int) -> IndexedImage | None:
        """Retrieve an image embedding from the repository by its ID."""
        ...

    def query_images(self, query: ImageSearchQuery) -> list[IndexedImage]:
        """Query images based on the provided search query."""
        ...
