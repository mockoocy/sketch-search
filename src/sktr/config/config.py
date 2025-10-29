from pathlib import Path
import torch
import yaml
# konfiugracja jako model pydantic
from sktr.config.config_model import Config


def _load_config(file_path: Path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file) or {}
    return Config.model_validate(config_data)


_REPOSITORY_ROOT = Path(__file__).parent.parent.parent

CFG = _load_config(_REPOSITORY_ROOT / "config.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
