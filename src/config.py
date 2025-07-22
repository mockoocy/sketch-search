from dataclasses import dataclass


@dataclass
class Config:
    image_size: int = 224
    jigsaw_grid: int = 3
    batch_size: int = 256
    epochs: int = 100
    base_lr: float = 0.075
    use_jigsaw: bool = True
    jigsaw_weight: float = 0.5
    opt: str = "adamw"  # 'adam' | 'adamw'
    thinning_iters: int = 5  # for sketch thinning


CFG = Config()
