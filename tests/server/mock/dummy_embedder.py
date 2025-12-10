class DummyEmbedder:
    name = "dummy"

    def embed(self, images: bytearray) -> list[list[float]]:
        return [[idx * 0.1 for idx in range(4)] for _ in images]
