from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from xlmr_sentiment.config import Config, load_config
from xlmr_sentiment.data import load_processed, prepare_data
from xlmr_sentiment.metrics import classification_metrics, expected_calibration_error, softmax

ROOT = Path(__file__).resolve().parents[1]
def test_default_config_loads_with_overrides():
    cfg = load_config(ROOT / "configs/default.yaml", ["training.epochs=1", "model.max_length=64"])
    assert cfg.training.epochs == 1 and cfg.model.max_length == 64
    assert cfg.label2id == {"negative": 0, "neutral": 1, "positive": 2}
    assert {s.language for s in cfg.data.sources} == {"en", "ur", "ur-Latn"}
def test_config_rejects_unknown_label_mapping():
    with pytest.raises(ValidationError):
        Config.model_validate({
            "labels": ["negative", "positive"],
            "data": {"sources": [{"name": "x", "local_path": "a.csv", "language": "en",
                                  "label_map": {"1": "very_positive"}}]},
        })

def test_config_requires_exactly_one_location():
    with pytest.raises(ValidationError):
        Config.model_validate({"labels": ["a", "b"], "data": {"sources": [
            {"name": "x", "language": "en", "label_map": {"a": "a"}}]}})

def test_prepare_data_splits_without_leakage(tmp_path):
    cfg = load_config(ROOT / "configs/smoke.yaml", [f"data.processed_dir={tmp_path.as_posix()}"])
    for s in cfg.data.sources:
        s.local_path = str(ROOT / s.local_path)
    splits = prepare_data(cfg)
    for name in ("train", "validation", "test"):
        df = load_processed(cfg, name)
        assert len(df) == len(splits[name]) > 0
        assert set(df.columns) >= {"text", "label", "label_id", "language", "source"}
        assert df["label_id"].between(0, 2).all()
    tr = set(splits["train"]["text"].str.lower())
    assert not tr & set(splits["test"]["text"].str.lower())
    assert not tr & set(splits["validation"]["text"].str.lower())
    assert set(splits["train"]["language"]) == {"en", "ur", "ur-Latn"}
    assert (tmp_path / "stats.json").exists()

def test_conflicting_duplicates_are_dropped(tmp_path):
    rows = [("great product", "positive"), ("great product", "negative")] + \
           [(f"text number {i}", ["negative", "neutral", "positive"][i % 3]) for i in range(60)]
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(tmp_path / "d.csv", index=False)
    cfg = Config.model_validate({
        "labels": ["negative", "neutral", "positive"],
        "data": {"processed_dir": str(tmp_path / "out"), "sources": [{
            "name": "d", "local_path": str(tmp_path / "d.csv"), "language": "en",
            "label_map": {"negative": "negative", "neutral": "neutral", "positive": "positive"}}]},
    })
    splits = prepare_data(cfg)
    all_text = pd.concat(splits.values())["text"]
    assert "great product" not in set(all_text)

def test_metrics_and_calibration():
    y = np.array([0, 1, 2, 2, 0, 1])
    p = np.array([0, 1, 2, 1, 0, 1])
    m = classification_metrics(y, p, ["neg", "neu", "pos"], ["en", "en", "en", "ur", "ur", "ur"])
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert set(m["by_group"]) == {"en", "ur"} and m["by_group"]["en"]["accuracy"] == 1.0
    perfect = np.eye(3)[y]
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)
    assert np.allclose(softmax(np.array([[1.0, 1.0]])), 0.5)
