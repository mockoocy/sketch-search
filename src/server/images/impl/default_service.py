from pathlib import Path

from server.images.errors import ImageNotFoundError
from server.images.models import ImageSearchQuery
from server.images.repository import ImageRepository
from server.index.embedder import Embedder
from server.index.models import IndexedImage
from server.index.repository import IndexedImageRepository


class DefaultImageService:
    def __init__(
        self,
        image_repository: ImageRepository,
        indexed_image_repository: IndexedImageRepository,
        embedder: Embedder,
    ) -> None:
        self._image_repository = image_repository
        self._indexed_image_repository = indexed_image_repository
        self._embedder = embedder

    def add_image(self, image: bytes, relative_path: Path) -> None:
        # It is assumed that validation is handled in the repository layer
        # indexing the image will be detected by fs watcher.
        self._image_repository.add_image_file(image, relative_path)

    def remove_image(self, image_id: int) -> None:
        indexed_image = self._indexed_image_repository.get_image_by_id(image_id)
        if indexed_image:
            img_path = Path(indexed_image.path)
            self._image_repository.remove_image_file(img_path)
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
