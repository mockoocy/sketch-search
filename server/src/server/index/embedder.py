from typing import Protocol

import numpy as np
import numpy.typing as npt

from server.index.models import Embedding


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

    def embed(self, images: npt.NDArray[np.float32]) -> Embedding:
        """
        Embed a batch of images.

        Args:
            images: A list of bytes objects representing a batch of images.

        Returns:
            A list of lists of floats representing the embeddings for each image.
        """
        ...
