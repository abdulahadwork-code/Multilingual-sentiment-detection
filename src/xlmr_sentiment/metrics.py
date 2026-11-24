from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], groups: Sequence[str] | None = None
) -> dict:
    """Overall + per-group metrics. Labels absent from a group are excluded from that group's macro-F1."""
    ids = list(range(len(labels)))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=ids, target_names=list(labels), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "n": int(len(y_true)),
    }
    if groups is not None:
        groups = np.asarray(groups)
        out["by_group"] = {}
        for g in sorted({str(x) for x in groups}):
            m = groups == g
            present = sorted(set(y_true[m].tolist()))
            out["by_group"][g] = {
                "n": int(m.sum()),
                "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
                "macro_f1": float(f1_score(y_true[m], y_pred[m], labels=present, average="macro", zero_division=0)),
                "labels_present": [labels[i] for i in present],
            }
    return out
