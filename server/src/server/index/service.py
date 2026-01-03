from pathlib import Path
from typing import Protocol


class IndexingService(Protocol):
    def embed_images(self, relative_paths: list[Path]) -> None:
        """Embed a batch of images located at the given paths into the index."""
        ...

    def remove_image(self, relative_path: Path) -> None:
        """Remove an image from the index by its Path."""
        ...

    def move_image(self, old_path: Path, new_path: Path) -> None:
        """Update the path of an image in the index."""
        ...

    def get_collection_size(self) -> int:
        """Get the total number of indexed images in the collection."""
        ...

    def reindex_images_with_different_model(self, model_name: str) -> None:
        """Reindex images that were indexed with a different model than specified."""
        ...
