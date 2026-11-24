from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config
from .data import load_processed
from .inference import SentimentPredictor
from .metrics import classification_metrics, expected_calibration_error, softmax
from .utils import write_json

log = logging.getLogger(__name__)


def predict_split(cfg: Config, model_dir: str | Path, split: str = "test") -> pd.DataFrame:
    df = load_processed(cfg, split).reset_index(drop=True)
    predictor = SentimentPredictor(model_dir, device=cfg.serving.device, quantize_cpu=False,
                                   batch_size=cfg.training.eval_batch_size)
    if predictor.labels != cfg.labels:
        raise ValueError(f"model labels {predictor.labels} != config labels {cfg.labels}")
    logits = predictor.predict_logits(df["text"].tolist())
    probs = softmax(logits)
    df["pred_id"] = probs.argmax(1)
    df["pred"] = [cfg.labels[i] for i in df["pred_id"]]
    df["confidence"] = probs.max(1)
    for i, lbl in enumerate(cfg.labels):
        df[f"p_{lbl}"] = probs[:, i]
    df["correct"] = df["pred_id"] == df["label_id"]
    return df

def evaluate(cfg: Config, model_dir: str | Path, split: str = "test", out_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir or Path(cfg.training.output_dir) / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = predict_split(cfg, model_dir, split)
    probs = df[[f"p_{lbl}" for lbl in cfg.labels]].to_numpy()
    y = df["label_id"].to_numpy()
    metrics = classification_metrics(y, df["pred_id"].to_numpy(), cfg.labels, df["language"].tolist())
    metrics["ece"] = expected_calibration_error(probs, y, cfg.analysis.calibration_bins)
    metrics["split"] = split
    df.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
    df.drop(columns=["label_id", "pred_id"]).to_csv(out_dir / f"predictions_{split}.csv", index=False,
                                                    encoding="utf-8-sig")  
    write_json(metrics, out_dir / f"metrics_{split}.json")
    log.info("%s: accuracy=%.4f macro_f1=%.4f ece=%.4f | per-language macro-F1: %s", split, metrics["accuracy"],
             metrics["macro_f1"], metrics["ece"],
             {g: round(v["macro_f1"], 4) for g, v in metrics["by_group"].items()})
    return metrics

def majority_baseline(cfg: Config, split: str = "test") -> dict:
    """Reference point for the report: always predict the most frequent training label."""
    train = load_processed(cfg, "train")
    test = load_processed(cfg, split)
    majority = int(train["label_id"].mode()[0])
    preds = np.full(len(test), majority)
    return classification_metrics(test["label_id"].to_numpy(), preds, cfg.labels, test["language"].tolist())
