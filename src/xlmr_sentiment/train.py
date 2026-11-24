from __future__ import annotations

import inspect
import logging
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers import __version__ as hf_version

from . import __version__
from .config import Config
from .data import load_processed
from .metrics import classification_metrics
from .utils import set_seed, write_json

log = logging.getLogger(__name__)

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._class_weights.to(outputs.logits.device) if self._class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self._label_smoothing)
        loss = loss_fn(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss

def inverse_frequency_weights(label_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = np.clip(w / w.mean(), 0.25, 4.0)  # keep weights sane for very rare classes
    return torch.tensor(w, dtype=torch.float32)

def _training_args(**kwargs) -> TrainingArguments:
    """Build TrainingArguments portably across transformers 4.4x and 5.x (length-grouped batching moved)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "group_by_length" in params:
        kwargs["group_by_length"] = True
    elif "train_sampling_strategy" in params:
        kwargs["train_sampling_strategy"] = "group_by_length"
    return TrainingArguments(**kwargs)

def _to_hf(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
                  batched=True, remove_columns=["text"], desc="tokenising")

def train(cfg: Config) -> Path:
    set_seed(cfg.seed)
    tcfg = cfg.training
    out_dir = Path(tcfg.output_dir)

    train_df = load_processed(cfg, "train")
    val_df = load_processed(cfg, "validation")
    if tcfg.max_train_samples:
        train_df = train_df.sample(n=min(tcfg.max_train_samples, len(train_df)), random_state=cfg.seed)
    if tcfg.max_eval_samples:
        val_df = val_df.sample(n=min(tcfg.max_eval_samples, len(val_df)), random_state=cfg.seed)
    val_df = val_df.reset_index(drop=True)
    log.info("train=%d validation=%d", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model_config = AutoConfig.from_pretrained(
        cfg.model.name,
        num_labels=len(cfg.labels),
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        hidden_dropout_prob=cfg.model.dropout,
        attention_probs_dropout_prob=cfg.model.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, config=model_config)
    train_ds = _to_hf(train_df, tokenizer, cfg.model.max_length)
    val_ds = _to_hf(val_df, tokenizer, cfg.model.max_length)
    val_langs = val_df["language"].to_numpy()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.asarray(logits).argmax(-1)
        m = classification_metrics(np.asarray(labels), preds, cfg.labels, val_langs)
        flat = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        flat.update({f"macro_f1_{g}": v["macro_f1"] for g, v in m["by_group"].items()})
        return flat

    use_fp16 = torch.cuda.is_available() if tcfg.fp16 == "auto" else bool(tcfg.fp16)
    n_gpus = max(torch.cuda.device_count(), 1)
    steps_per_epoch = math.ceil(len(train_ds) / (tcfg.train_batch_size * n_gpus * tcfg.gradient_accumulation_steps))
    warmup_steps = int(math.ceil(steps_per_epoch * tcfg.epochs * tcfg.warmup_ratio))
    args = _training_args(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=tcfg.epochs,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=tcfg.train_batch_size,
        per_device_eval_batch_size=tcfg.eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=tcfg.eval_steps,
        save_steps=tcfg.eval_steps,
        logging_steps=tcfg.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
    )
    class_weights = (
        inverse_frequency_weights(train_df["label_id"].to_numpy(), len(cfg.labels)) if tcfg.class_weighting else None
    )
    if class_weights is not None:
        log.info("class weights: %s", dict(zip(cfg.labels, class_weights.tolist(), strict=True)))

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tcfg.early_stopping_patience)],
        class_weights=class_weights,
        label_smoothing=tcfg.label_smoothing,
    )
    train_result = trainer.train()
    val_metrics = trainer.evaluate()

    best_dir = out_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    write_json(
        {
            "package_version": __version__,
            "transformers_version": hf_version,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "base_model": cfg.model.name,
            "max_length": cfg.model.max_length,
            "labels": cfg.labels,
            "train_rows": len(train_df),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "train_metrics": train_result.metrics,
            "validation_metrics": val_metrics,
            "config": cfg.model_dump(),
        },
        best_dir / "training_meta.json",
    )
    write_json(trainer.state.log_history, out_dir / "log_history.json")
    log.info("best model saved to %s | val macro-F1 = %.4f", best_dir, val_metrics.get("eval_macro_f1", float("nan")))
    return best_dir
