from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import load_config
from .utils import setup_logging

log = logging.getLogger("xlmr_sentiment")

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", default="configs/default.yaml", help="YAML config file")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set training.epochs=1 (repeatable)")

def _model_dir(args, cfg) -> str:
    return args.model_dir or str(Path(cfg.training.output_dir) / "best")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xlmr-sentiment", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("prepare-data", "train", "pipeline"):
        _common(sub.add_parser(name))
    for name in ("evaluate", "analyze"):
        p = sub.add_parser(name)
        _common(p)
        p.add_argument("--model-dir", help="trained model directory (default: <training.output_dir>/best)")
        p.add_argument("--split", default="test", choices=["validation", "test"])
    p = sub.add_parser("predict")
    _common(p)
    p.add_argument("--model-dir")
    p.add_argument("texts", nargs="*", help="texts to classify (reads stdin lines if omitted)")
    p = sub.add_parser("serve")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("push-to-hub", help="upload the trained model (+ metrics) to the Hugging Face Hub")
    p.add_argument("repo_id", help="e.g. your-username/xlmr-urdu-english-sentiment")
    p.add_argument("--model-dir", default="artifacts/xlmr-sentiment/best")
    p.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.cmd == "push-to-hub":
        from .hub import push_to_hub
        print(push_to_hub(args.model_dir, args.repo_id, private=args.private))
        return 0
    if args.cmd == "serve":
        import uvicorn
        if args.model_dir:
            os.environ["MODEL_DIR"] = args.model_dir
        uvicorn.run("xlmr_sentiment.api.app:app_factory", factory=True, host=args.host, port=args.port,
                    workers=args.workers, log_config=None)
        return 0
    cfg = load_config(args.config, args.overrides)
    if args.cmd in ("prepare-data", "pipeline"):
        from .data import prepare_data
        prepare_data(cfg)
    if args.cmd in ("train", "pipeline"):
        from .train import train
        train(cfg)
    if args.cmd in ("evaluate", "analyze", "pipeline"):
        from .error_analysis import run_error_analysis
        from .evaluate import evaluate, majority_baseline
        model_dir = _model_dir(args, cfg) if args.cmd != "pipeline" else str(Path(cfg.training.output_dir) / "best")
        split = getattr(args, "split", "test")
        eval_dir = Path(cfg.training.output_dir) / "eval"
        pred_path = eval_dir / f"predictions_{split}.parquet"
        if args.cmd in ("evaluate", "pipeline") or not pred_path.exists():
            evaluate(cfg, model_dir, split, eval_dir)
        if args.cmd in ("analyze", "pipeline"):
            report = run_error_analysis(cfg, pred_path, model_dir, baseline=majority_baseline(cfg, split))
            print(f"\nError analysis report: {report}")
    if args.cmd == "predict":
        from .inference import SentimentPredictor
        texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
        predictor = SentimentPredictor(_model_dir(args, cfg), device=cfg.serving.device,
                                       quantize_cpu=cfg.serving.quantize_cpu)
        for text, pred in zip(texts, predictor.predict(texts), strict=True):
            print(json.dumps({"text": text, **pred.to_dict()}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
