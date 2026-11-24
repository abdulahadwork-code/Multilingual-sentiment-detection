from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Language = Literal["en", "ur", "ur-Latn"]

class SourceConfig(BaseModel):
    name: str
    language: Language
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, str]
    hf_path: str | None = None
    hf_config: str | None = None
    local_path: str | None = None  
    max_per_split: dict[str, int] = Field(default_factory=dict)
    @model_validator(mode="after")
    def _one_location(self) -> SourceConfig:
        if bool(self.hf_path) == bool(self.local_path):
            raise ValueError(f"source '{self.name}': set exactly one of hf_path or local_path")
        return self

class DataConfig(BaseModel):
    processed_dir: str = "data/processed"
    val_size: float = Field(0.1, gt=0, lt=0.5)
    test_size: float = Field(0.1, gt=0, lt=0.5)
    min_chars: int = 3
    max_chars: int = 2000
    dedupe: bool = True
    sources: list[SourceConfig]

class ModelConfig(BaseModel):
    name: str = "FacebookAI/xlm-roberta-base"
    max_length: int = Field(160, ge=8, le=512)
    dropout: float = Field(0.1, ge=0, lt=1)

class TrainingConfig(BaseModel):
    output_dir: str = "artifacts/xlmr-sentiment"
    epochs: float = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    class_weighting: bool = True
    label_smoothing: float = 0.0
    early_stopping_patience: int = 2
    fp16: bool | Literal["auto"] = "auto"
    eval_steps: int = 500
    logging_steps: int = 50
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

class AnalysisConfig(BaseModel):
    output_dir: str = "reports/error_analysis"
    top_k_examples: int = 25
    calibration_bins: int = 10

class ServingConfig(BaseModel):
    model_dir: str = "artifacts/xlmr-sentiment/best"
    max_batch_size: int = 64
    max_text_chars: int = 5000
    device: str = "auto"
    quantize_cpu: bool = False

class Config(BaseModel):
    seed: int = 42
    labels: list[str]
    data: DataConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    serving: ServingConfig = ServingConfig()

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, v: list[str]) -> list[str]:
        if len(v) < 2 or len(set(v)) != len(v):
            raise ValueError("labels must contain at least two unique names")
        return v

    @model_validator(mode="after")
    def _label_maps_valid(self) -> Config:
        for src in self.data.sources:
            bad = set(src.label_map.values()) - set(self.labels)
            if bad:
                raise ValueError(f"source '{src.name}' maps to unknown labels {sorted(bad)}")
        return self

    @property
    def label2id(self) -> dict[str, int]:
        return {lbl: i for i, lbl in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

def _parse_scalar(raw: str) -> Any:
    return yaml.safe_load(raw)

def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override '{key}': '{p}' is not a mapping")
        node[parts[-1]] = _parse_scalar(raw)
    return cfg

def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(apply_overrides(raw, overrides))
