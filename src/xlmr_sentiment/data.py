from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import Config, SourceConfig
from .text import normalize
from .utils import write_json

log = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
_SPLIT_ALIASES = {"train": "train", "validation": "validation", "valid": "validation", "dev": "validation",
                  "val": "validation", "test": "test"}

def _load_raw(src: SourceConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if src.hf_path:
        from datasets import load_dataset
        ds = load_dataset(src.hf_path, src.hf_config)
        for split_name, split in ds.items():
            canon = _SPLIT_ALIASES.get(split_name)
            if canon is None:
                log.warning("[%s] ignoring unknown split '%s'", src.name, split_name)
                continue
            feat = split.features.get(src.label_column)
            df = split.select_columns([src.text_column, src.label_column]).to_pandas()
            names = getattr(feat, "names", None)
            if names:  
                df[src.label_column] = df[src.label_column].map(lambda i, n=names: n[int(i)] if i >= 0 else None)
            frames[canon] = df
    else:
        path = Path(src.local_path)
        files = {s: next(iter(sorted(path.glob(f"{s}.*"))), None) for s in SPLITS} if path.is_dir() else {"train": path}
        for split_name, f in files.items():
            if f is None:
                continue
            reader = {".csv": pd.read_csv, ".parquet": pd.read_parquet,
                      ".jsonl": lambda p: pd.read_json(p, lines=True)}.get(f.suffix.lower())
            if reader is None:
                raise ValueError(f"[{src.name}] unsupported file type: {f}")
            frames[split_name] = reader(f)[[src.text_column, src.label_column]]
    if not frames:
        raise RuntimeError(f"[{src.name}] no usable splits found")
    return frames
def _harmonise(df: pd.DataFrame, src: SourceConfig, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame({
        "text": df[src.text_column].astype(str),
        "raw_label": df[src.label_column].astype(str).str.strip(),
    })
    out["label"] = out["raw_label"].map(src.label_map)
    unmapped = out["label"].isna().sum()
    if unmapped:
        log.warning("[%s] dropping %d rows with unmapped labels %s", src.name, unmapped,
                    sorted(out.loc[out["label"].isna(), "raw_label"].unique())[:10])
    out = out.dropna(subset=["label"]).drop(columns="raw_label")
    out["text"] = out["text"].map(lambda t: normalize(t, cfg.data.max_chars))
    out = out[out["text"].str.len() >= cfg.data.min_chars]
    out["language"] = src.language
    out["source"] = src.name
    return out.reset_index(drop=True)

def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    strat = df["label"] if df["label"].value_counts().min() >= 2 else None
    sampled, _ = train_test_split(df, train_size=n, stratify=strat, random_state=seed)
    return sampled

def _split_frame(df: pd.DataFrame, size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = df["label"] if df["label"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=size, stratify=strat, random_state=seed)

def build_source(src: SourceConfig, cfg: Config) -> dict[str, pd.DataFrame]:
    raw = _load_raw(src)
    splits = {k: _harmonise(v, src, cfg) for k, v in raw.items()}
    if src.max_per_split.get("train") and "train" in splits:
        pool_cap = src.max_per_split["train"]
        missing = sum(s not in splits for s in ("validation", "test"))
        splits["train"] = _stratified_sample(splits["train"], int(pool_cap * (1 + 0.15 * missing)), cfg.seed)
    if "test" not in splits:
        splits["train"], splits["test"] = _split_frame(splits["train"], cfg.data.test_size, cfg.seed)
    if "validation" not in splits:
        splits["train"], splits["validation"] = _split_frame(splits["train"], cfg.data.val_size, cfg.seed)
    for split, cap in src.max_per_split.items():
        if split in splits:
            splits[split] = _stratified_sample(splits[split], cap, cfg.seed)
    log.info("[%s] %s", src.name, {k: len(v) for k, v in splits.items()})
    return splits

def _dedupe_and_deleak(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove exact duplicates, label-conflicting duplicates, and any train/val text that also appears in test."""
    everything = pd.concat(splits.values(), ignore_index=True)
    n_labels = everything.assign(_k=everything["text"].str.lower()).groupby("_k")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    if conflicting:
        log.warning("dropping %d texts that carry conflicting labels across the corpus", len(conflicting))
    for name in SPLITS:
        df = splits[name]
        df = df[~df["text"].str.lower().isin(conflicting)]
        splits[name] = df.loc[~df["text"].str.lower().duplicated()]
    test_keys = set(splits["test"]["text"].str.lower())
    val = splits["validation"]
    leaked_val = val["text"].str.lower().isin(test_keys)
    splits["validation"] = val[~leaked_val]
    held_out = test_keys | set(splits["validation"]["text"].str.lower())
    tr = splits["train"]
    leaked_tr = tr["text"].str.lower().isin(held_out)
    splits["train"] = tr[~leaked_tr]
    log.info("de-leak: removed %d train and %d validation rows overlapping held-out splits",
             leaked_tr.sum(), leaked_val.sum())
    return splits

def describe(splits: dict[str, pd.DataFrame]) -> dict:
    stats = {}
    for name, df in splits.items():
        stats[name] = {
            "rows": len(df),
            "by_label": df["label"].value_counts().to_dict(),
            "by_language": df["language"].value_counts().to_dict(),
            "by_language_label": {f"{lang}/{lbl}": int(n) for (lang, lbl), n in
                                  df.groupby(["language", "label"]).size().items()},
            "mean_chars": round(float(df["text"].str.len().mean()), 1) if len(df) else 0.0,
        }
    return stats

def prepare_data(cfg: Config) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS}
    for src in cfg.data.sources:
        for split, df in build_source(src, cfg).items():
            parts[split].append(df)
    splits = {s: pd.concat(parts[s], ignore_index=True) for s in SPLITS}
    if cfg.data.dedupe:
        splits = _dedupe_and_deleak(splits)
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        df["label_id"] = df["label"].map(cfg.label2id).astype("int64")
        splits[name] = df
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    stats = describe(splits)
    write_json(stats, out_dir / "stats.json")
    log.info("saved processed data to %s: %s", out_dir, {k: v["rows"] for k, v in stats.items()})
    return splits

def load_processed(cfg: Config, split: str) -> pd.DataFrame:
    path = Path(cfg.data.processed_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `xlmr-sentiment prepare-data` first")
    return pd.read_parquet(path)
