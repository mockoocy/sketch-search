import numpy as np
import numpy.typing as npt


class DummyEmbedder:
    name = "dummy"

    def embed(self, images: list[bytes]) -> npt.NDArray[np.float32]:
        return np.array(
            [[0.1 * idx for idx in range(1536)] for _ in range(len(images))],
            dtype=np.float32,
        )
