import json
from pathlib import Path
import pytest
from xlmr_sentiment.config import load_config

ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.slow
def test_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from xlmr_sentiment.data import prepare_data
    from xlmr_sentiment.error_analysis import run_error_analysis
    from xlmr_sentiment.evaluate import evaluate, majority_baseline
    from xlmr_sentiment.inference import SentimentPredictor
    from xlmr_sentiment.train import train

    cfg = load_config(ROOT / "configs/smoke.yaml", [
        f"data.processed_dir={(tmp_path / 'data').as_posix()}",
        f"training.output_dir={(tmp_path / 'model').as_posix()}",
        f"analysis.output_dir={(tmp_path / 'report').as_posix()}",
    ])
    for s in cfg.data.sources:
        s.local_path = str(ROOT / s.local_path)

    prepare_data(cfg)
    try:
        best = train(cfg)
    except OSError as e:  # offline CI
        pytest.skip(f"could not download tiny model: {e}")
    meta = json.loads((best / "training_meta.json").read_text(encoding="utf-8"))
    assert meta["labels"] == cfg.labels

    metrics = evaluate(cfg, best, "test", tmp_path / "eval")
    assert 0.0 <= metrics["macro_f1"] <= 1.0 and set(metrics["by_group"]) == {"en", "ur", "ur-Latn"}

    report = run_error_analysis(cfg, tmp_path / "eval" / "predictions_test.parquet", best,
                                baseline=majority_baseline(cfg))
    text = report.read_text(encoding="utf-8")
    assert "Key findings" in text and "Slice analysis" in text
    assert (report.parent / "figures" / "calibration.png").exists()
    assert (report.parent / "manual_annotation_sheet.csv").exists()

    pred = SentimentPredictor(best, device="cpu", quantize_cpu=True)
    out = pred.predict(["great phone", "یہ فون بہت برا ہے", "bohat acha hai"])
    assert len(out) == 3 and all(abs(sum(p.scores.values()) - 1) < 1e-3 for p in out)
    assert out[1].language == "ur"
