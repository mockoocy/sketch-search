from pathlib import Path
from typing import Protocol


class IndexingService(Protocol):
    def embed_images(self, image_paths: list[Path]) -> None:
        """Embed a batch of images located at the given paths into the index."""
        ...

    def remove_image(self, image_path: Path) -> None:
        """Remove an image from the index by its Path."""
        ...

    def move_image(self, old_path: Path, new_path: Path) -> None:
        """Update the path of an image in the index."""
        ...
