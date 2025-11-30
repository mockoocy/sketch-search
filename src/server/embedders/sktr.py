import torch

from sktr.model import Embedder as SktrModel
from sktr.model import TimmBackbone

WEIGHTS_PATH = "../sktr.pth"


class SktrEmbedder:
    def __init__(
        self,
        embedding_size: int = 512,
        hidden_layer_size: int = 2048,
        timm_encoder_name: str = "resnet18",
    ) -> None:
        backbone = TimmBackbone(name=timm_encoder_name, pretrained=False)
        self.model = SktrModel(
            backbone=backbone,
            embedding_size=embedding_size,
            hidden_layer_size=hidden_layer_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict=state_dict)

    def embed(self, images: bytearray) -> list[list[float]]:
        return self.model.embed(images)
