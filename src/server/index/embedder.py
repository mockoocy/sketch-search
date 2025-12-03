from typing import Protocol


class Embedder(Protocol):
    """
    Class used to create embeddings for images.

    Overriding `embed` method is meant to create embeddings from a
    bytes form of the image.

    This is a framework-agnostic interface. Implementations can use
    any deep learning framework (e.g., JAX, PyTorch) to create
    the embeddings.
    """

    name: str

    def embed(self, images: list[bytes]) -> list[list[float]]:
        """
        Embed a batch of images.

        Args:
            images: A list of bytes objects representing a batch of images.

        Returns:
            A list of lists of floats representing the embeddings for each image.
        """
        ...
