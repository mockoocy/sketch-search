from typing import Literal

from pydantic import BaseModel, Field


class TrainingSettings(BaseModel):
    epochs: int = Field(default=1, gt=0, le=1000)
    base_lr: float = Field(default=5e-3, gt=0, le=100)
    batch_size: int = Field(default=32, ge=1)
    optimizer: Literal["adam", "adamw"] = "adam"
    sketches_path: str = "./data/sketches/"
    images_path: str = "./data/images/"
    dcl_temperature: float = Field(default=0.2, gt=0.0, le=1.0)
    fraction_of_samples: float = Field(default=1.0, gt=0.0, le=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    model_save_path: str = "./models/"


class ValidationSettings(BaseModel):
    validation_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    eval_every_steps: int = Field(default=100, gt=0)


class SkitterSettings(BaseModel):
    projection_head_size: int = Field(default=256, gt=0, le=4096)
    embedding_size: int = Field(default=256, gt=0, le=4096)
    encoder_name: str = "resnet50"


class Config(BaseModel):
    skitter: SkitterSettings = SkitterSettings()
    training: TrainingSettings = TrainingSettings()
    validation: ValidationSettings = ValidationSettings()
