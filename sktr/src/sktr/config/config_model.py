from typing import Literal

from pydantic import BaseModel, Field


class TrainingPhaseSettings(BaseModel):
    sketches_path: str = Field(default="./data/unsupervised/sketches/")
    images_path: str = Field(default="./data/unsupervised/images/")
    temperature: float = Field(default=0.1, gt=0.0, le=1.0)
    fraction_of_samples: float = Field(default=1.0, gt=0.0, le=1.0)
    epochs: int = Field(default=1, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    base_lr: float = Field(default=1e-3, gt=0.0)
    min_lr_ratio: float = Field(default=0.1, gt=0.0, le=1.0)


class TrainingPhase2Settings(TrainingPhaseSettings):
    samples_per_class: int = Field(default=4, ge=1)


class EarlyStopConfig(BaseModel):
    monitor: str = "mAP@10"
    mode: str = "max"
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    warmup_evals: int = Field(default=1, ge=0)


class TrainingSettings(BaseModel):
    batch_size: int = Field(default=32, ge=1)
    optimizer: Literal["adam", "adamw"] = "adam"
    # shall remove test fraction in the future.
    test_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    model_save_path: str = "./models/"
    phase_1: TrainingPhaseSettings = TrainingPhaseSettings()
    phase_2: TrainingPhase2Settings = TrainingPhase2Settings(
        sketches_path="./data/supervised/sketches/",
        images_path="./data/supervised/images/",
        temperature=0.07,
        fraction_of_samples=1.0,
    )
    num_workers: int = Field(default=6, ge=0)
    early_stopping: EarlyStopConfig = EarlyStopConfig()


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
