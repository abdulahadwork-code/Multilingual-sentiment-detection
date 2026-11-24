from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CARD = """---
language: [en, ur]
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: {base_model}
tags: [sentiment-analysis, urdu, roman-urdu, xlm-roberta, multilingual]
---

# {repo_name}

XLM-RoBERTa fine-tuned for 3-class sentiment (negative / neutral / positive) on English, Urdu
(Arabic script) and Roman Urdu.

| metric (validation) | value |
|---|---|
{metric_rows}

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo_id}")
clf(["yeh movie bohat achi thi", "یہ فلم بہت بری تھی", "The service was okay"])
```

Note: the training pipeline normalises text (Urdu character unification, URL/mention placeholders).
For best results use the same `normalize()` function from the project before calling the model.
"""

def push_to_hub(model_dir: str | Path, repo_id: str, private: bool = False) -> str:
    from huggingface_hub import HfApi

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"{model_dir} has no config.json - train a model first")
    meta_path = model_dir / "training_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    val = meta.get("validation_metrics", {})
    rows = "\n".join(f"| {k.removeprefix('eval_')} | {v:.4f} |" for k, v in val.items()
                     if k.startswith("eval_") and ("f1" in k or "accuracy" in k)) or "| – | – |"
    (model_dir / "README.md").write_text(
        _CARD.format(base_model=meta.get("base_model", "FacebookAI/xlm-roberta-base"), repo_name=repo_id.split("/")[-1],
                     repo_id=repo_id, metric_rows=rows),
        encoding="utf-8",
    )
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(model_dir), commit_message="Upload fine-tuned model",
                      ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "scheduler*", "rng_state*"])
    url = f"https://huggingface.co/{repo_id}"
    log.info("uploaded %s to %s", model_dir, url)
    return url
