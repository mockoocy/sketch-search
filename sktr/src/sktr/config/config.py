from pathlib import Path
import os
import torch
import yaml
import argparse
from sktr.config.config_model import Config


def _load_config(file_path: Path) -> Config:
    with Path.open(file_path) as file:
        config_data = yaml.safe_load(file) or {}
    return Config.model_validate(config_data)

_REPOSITORY_ROOT = Path(__file__).parent.parent.parent.parent


def load_cfg_from_cli() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("SKTR_CONFIG", str(_REPOSITORY_ROOT / "config.yaml")),
        help="Path to config file",
    )
    args, _ = parser.parse_known_args()
    return _load_config(Path(args.config))




CFG = load_cfg_from_cli()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
