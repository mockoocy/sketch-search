import numpy as np
import numpy.typing as npt
import torch

from server.index.models import Embedding
from server.logger import app_logger
from sktr.model import Embedder as SktrModel
from sktr.model import TimmBackbone


class SktrEmbedder:
    name = "sktr-convnext-tiny"

    def __init__(
        self,
        embedding_size: int = 1536,
        hidden_layer_size: int = 2048,
        timm_encoder_name: str = "convnext_tiny.fb_in22k",
        weights_path: str | None = None,
    ) -> None:
        backbone = TimmBackbone(name=timm_encoder_name, pretrained=True)
        self.model = SktrModel(
            backbone=backbone,
            embedding_size=embedding_size,
            hidden_layer_size=hidden_layer_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app_logger.info("Loading SKTR embedder on device: %s", device)
        if weights_path:
            app_logger.info("Loading SKTR weights from: %s", weights_path)
            state_dict = torch.load(
                weights_path,
                map_location=device,
                weights_only=True,
            )
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict=state_dict)

    def embed(self, images: npt.NDArray[np.float32]) -> Embedding:
        app_logger.info("Embedding %d images using SKTR embedder...", len(images))
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        images_torch = torch.from_numpy(images).float().permute(0, 3, 1, 2)
        embedding = self.model.embed_photo(images_torch)
        return embedding.cpu().detach().numpy()
