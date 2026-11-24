from __future__ import annotations
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .metrics import softmax
from .text import normalize, profile_script
from .utils import resolve_device

log = logging.getLogger(__name__)

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def resolve_model_dir(name_or_path: str | Path, token: str | None = None) -> Path:
    path = Path(name_or_path)
    if (path / "config.json").exists():
        return path
    name = str(name_or_path)
    if path.is_absolute() or path.exists() or "\\" in name or not _REPO_ID_RE.match(name):
        raise FileNotFoundError(f"no model found at {name} (expected config.json) - train first")
    from huggingface_hub import list_repo_files, snapshot_download
    token = token or os.getenv("HF_TOKEN") or None
    files = list_repo_files(name, token=token)
    weights = [f for f in files if f.endswith(".safetensors")] or [f for f in files if f == "pytorch_model.bin"
                                                                    or f.startswith("pytorch_model-")]
    keep = [f for f in files if f.endswith((".json", ".model", ".txt")) and "/" not in f] + weights
    log.info("downloading %d files from the Hugging Face Hub repo %s", len(keep), name)
    return Path(snapshot_download(name, allow_patterns=keep, token=token))

@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]
    language: str
    code_mixed: bool

    def to_dict(self) -> dict:
        return asdict(self)

class SentimentPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        max_length: int | None = None,
        quantize_cpu: bool = False,
        batch_size: int = 32,
        max_chars: int = 5000,
    ):
        self.source = str(model_dir)
        model_dir = resolve_model_dir(model_dir)
        self.model_dir = model_dir
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_chars = max_chars
        meta_path = model_dir / "training_meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.max_length = max_length or self.meta.get("max_length", 256)

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        self.quantized = False
        if self.device == "cpu" and quantize_cpu:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            self.quantized = True
        self.model = model.to(self.device)
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        self._lock = threading.Lock()  # tokenizers + torch modules are not guaranteed re-entrant
        log.info("loaded model from %s on %s (quantized=%s) in %.1fs", model_dir, self.device, self.quantized,
                 time.perf_counter() - t0)
    @torch.inference_mode()
    def predict_logits(self, texts: list[str]) -> np.ndarray:
        """Return raw logits for already-normalised texts, preserving input order."""
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])  # length-sorted batches = far less padding
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        with self._lock:
            for start in range(0, len(texts), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt").to(self.device)
                out[idx] = self.model(**enc).logits.float().cpu().numpy()
        return out

    def predict(self, texts: list[str]) -> list[Prediction]:
        clean = [normalize(t, self.max_chars) for t in texts]
        probs = softmax(self.predict_logits(clean))
        results = []
        for raw, p in zip(texts, probs, strict=True):
            prof = profile_script(raw)
            k = int(p.argmax())
            results.append(Prediction(
                label=self.labels[k],
                confidence=round(float(p[k]), 4),
                scores={lbl: round(float(s), 4) for lbl, s in zip(self.labels, p, strict=True)},
                language=prof.language,
                code_mixed=prof.code_mixed,
            ))
        return results

    def warmup(self) -> None:
        self.predict(["warm up", "یہ ایک ٹیسٹ ہے", "yeh acha hai"])
