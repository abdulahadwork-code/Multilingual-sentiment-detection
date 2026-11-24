from __future__ import annotations
import logging
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
from sklearn.metrics import f1_score  
from .config import Config  
from .metrics import classification_metrics, expected_calibration_error  
from .text import has_negation, profile_script  
from .utils import write_json  

log = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
LANG_NAMES = {"en": "English", "ur": "Urdu (Arabic script)", "ur-Latn": "Roman Urdu"}

CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|yet|lekin|magar|par|phir bhi)\b|لیکن|مگر|البتہ|پھر بھی",
                         re.IGNORECASE)
SARCASM_RE = re.compile(r"😂|🙄|😏|🤣|😒|\b(?:yeah right|sure|wah|wow|great job|kya baat hai|bohat khoob)\b|واہ",
                        re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")
INTENSIFIER_RE = re.compile(r"\b(?:very|so|too|extremely|really|bohat|bahut|bht|boht|bilkul|zyada)\b|بہت|بالکل|انتہائی",
                            re.IGNORECASE)

def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

def add_features(df: pd.DataFrame, tokenizer=None, max_length: int | None = None) -> pd.DataFrame:
    df = df.copy()
    profiles = df["text"].map(profile_script)
    df["detected_language"] = profiles.map(lambda p: p.language)
    df["code_mixed"] = profiles.map(lambda p: p.code_mixed)
    df["n_words"] = df["text"].str.split().str.len()
    df["length_bucket"] = pd.cut(df["n_words"], [0, 5, 15, 40, 100, 10_000],
                                 labels=["1-5", "6-15", "16-40", "41-100", "100+"])
    df["has_negation"] = df["text"].map(has_negation)
    df["has_contrast"] = df["text"].str.contains(CONTRAST_RE)
    df["has_emoji"] = df["text"].str.contains(EMOJI_RE)
    df["sarcasm_cue"] = df["text"].str.contains(SARCASM_RE)
    df["has_intensifier"] = df["text"].str.contains(INTENSIFIER_RE)
    df["is_question"] = df["text"].str.contains(r"[?؟]")
    df["lang_id_mismatch"] = df["detected_language"] != df["language"]
    if tokenizer is not None and max_length:
        lens = [len(ids) for ids in tokenizer(df["text"].tolist(), add_special_tokens=True)["input_ids"]]
        df["n_tokens"] = lens
        df["truncated"] = df["n_tokens"] > max_length
    else:
        df["truncated"] = False
    return df

ERROR_CATEGORIES = {
    "negation": lambda r: r.has_negation,
    "contrast / mixed sentiment": lambda r: r.has_contrast,
    "possible sarcasm": lambda r: r.sarcasm_cue,
    "neutral boundary": lambda r: "neutral" in (r.label, r.pred),
    "polarity flip (pos<->neg)": lambda r: {r.label, r.pred} == {"positive", "negative"},
    "very short (<=5 words)": lambda r: r.n_words <= 5,
    "truncated input": lambda r: bool(r.truncated),
    "code-mixed": lambda r: r.code_mixed,
    "language-ID mismatch": lambda r: r.lang_id_mismatch,
    "confident error (label noise?)": lambda r: r.confidence >= 0.9,
}

def tag_errors(errors: pd.DataFrame) -> pd.DataFrame:
    errors = errors.copy()
    errors["auto_tags"] = errors.apply(
        lambda r: "; ".join(name for name, fn in ERROR_CATEGORIES.items() if fn(r)) or "other", axis=1
    )
    return errors

def slice_table(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    overall = df["correct"].mean()
    rows = []
    slices = {
        "all": pd.Series(True, index=df.index),
        **{f"language={lang}": df["language"] == lang for lang in sorted(df["language"].unique())},
        **{f"length={b}": df["length_bucket"] == b for b in df["length_bucket"].cat.categories},
        "negation": df["has_negation"],
        "contrast (but/lekin/magar)": df["has_contrast"],
        "possible sarcasm cue": df["sarcasm_cue"],
        "intensifier": df["has_intensifier"],
        "emoji": df["has_emoji"],
        "question": df["is_question"],
        "code-mixed": df["code_mixed"],
        "language-ID mismatch": df["lang_id_mismatch"],
        "truncated": df["truncated"],
    }
    for name, mask in slices.items():
        sub = df[mask]
        if len(sub) < 10:
            continue
        present = sorted(set(sub["label_id"].tolist()))
        rows.append({
            "slice": name, "n": len(sub), "share": len(sub) / len(df), "accuracy": sub["correct"].mean(),
            "macro_f1": f1_score(sub["label_id"], sub["pred_id"], labels=present, average="macro", zero_division=0),
            "delta_acc_vs_all": sub["correct"].mean() - overall,
        })
    return pd.DataFrame(rows)

def confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    err = df[~df["correct"]]
    pairs = err.groupby(["language", "label", "pred"]).size().rename("count").reset_index()
    totals = df.groupby("language").size()
    pairs["pct_of_language"] = pairs.apply(lambda r: r["count"] / totals[r["language"]], axis=1)
    return pairs.sort_values(["language", "count"], ascending=[True, False])

def calibration_curve(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = pd.cut(df["confidence"], edges, include_lowest=True)
    g = df.groupby(bins, observed=True).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                            confidence=("confidence", "mean"))
    return g.reset_index(drop=True)

def plot_confusions(df: pd.DataFrame, labels: list[str], path: Path) -> None:
    langs = sorted(df["language"].unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(4.2 * len(langs), 3.9), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, lang in zip(axes, langs, strict=True):
        sub = df[df["language"] == lang]
        cm = pd.crosstab(sub["label"], sub["pred"]).reindex(index=labels, columns=labels, fill_value=0).to_numpy()
        row = cm.sum(1, keepdims=True)
        norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
        ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, fontsize=8, color=INK_2)
        ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK_2)
        ax.set_xlabel("predicted", color=INK_2, fontsize=9)
        ax.set_ylabel("true", color=INK_2, fontsize=9)
        ax.set_title(f"{LANG_NAMES.get(lang, lang)}  (n={len(sub)})", fontsize=10, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices, row-normalised (recall per true class)", fontsize=11, color=INK, x=0.02,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

def plot_calibration(df: pd.DataFrame, n_bins: int, path: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="perfect calibration")
    for color, lang in zip(SERIES, sorted(df["language"].unique()), strict=False):
        c = calibration_curve(df[df["language"] == lang], n_bins)
        c = c[c["n"] >= 5]
        ax.plot(c["confidence"], c["accuracy"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LANG_NAMES.get(lang, lang))
    ax.set_xlabel("mean confidence in bin", color=INK_2, fontsize=9)
    ax.set_ylabel("accuracy in bin", color=INK_2, fontsize=9)
    ax.set_title("Reliability diagram", fontsize=10, color=INK, loc="left")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)

    _style_axes(ax2)
    bins = np.linspace(df["confidence"].min() if len(df) else 0, 1, 21)
    ax2.hist(df.loc[df["correct"], "confidence"], bins=bins, color=SERIES[0], alpha=0.9, label="correct",
             edgecolor=SURFACE, linewidth=1)
    ax2.hist(df.loc[~df["correct"], "confidence"], bins=bins, color=SERIES[1], alpha=0.9, label="error",
             edgecolor=SURFACE, linewidth=1)
    ax2.set_xlabel("confidence of predicted label", color=INK_2, fontsize=9)
    ax2.set_ylabel("examples", color=INK_2, fontsize=9)
    ax2.set_title("Confidence: correct vs. errors", fontsize=10, color=INK, loc="left")
    ax2.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

def plot_slices(slices: pd.DataFrame, overall_acc: float, path: Path) -> None:
    s = slices[slices["slice"] != "all"].sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.2), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.barh(s["slice"], s["accuracy"], color=SERIES[0], height=0.6)
    ax.axvline(overall_acc, color=INK_2, linestyle="--", linewidth=1)
    for y, (acc, n) in enumerate(zip(s["accuracy"], s["n"], strict=True)):
        ax.text(acc + 0.012, y, f"{acc:.1%}  n={n}", va="center", fontsize=7.5, color=INK_2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("accuracy", color=INK_2, fontsize=9)
    ax.set_title(f"Accuracy by data slice   (dashed line = overall {overall_acc:.1%})", fontsize=10, color=INK,
                 loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("–" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def _clip(t: str, n: int = 140) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"

def _findings(df: pd.DataFrame, slices: pd.DataFrame, tags: pd.Series, metrics: dict, baseline: dict) -> list[str]:
    out = []
    by_lang = metrics["by_group"]
    best = max(by_lang, key=lambda g: by_lang[g]["macro_f1"])
    worst = min(by_lang, key=lambda g: by_lang[g]["macro_f1"])
    out.append(f"Overall macro-F1 is **{metrics['macro_f1']:.3f}** vs. **{baseline['macro_f1']:.3f}** for the "
               f"majority-class baseline (accuracy {metrics['accuracy']:.1%} vs. {baseline['accuracy']:.1%}).")
    if best != worst:
        out.append(f"The model is strongest on **{LANG_NAMES.get(best, best)}** (macro-F1 "
                   f"{by_lang[best]['macro_f1']:.3f}) and weakest on **{LANG_NAMES.get(worst, worst)}** "
                   f"({by_lang[worst]['macro_f1']:.3f}), a gap of "
                   f"{by_lang[best]['macro_f1'] - by_lang[worst]['macro_f1']:.3f}.")
    s = slices[(slices["slice"] != "all") & (slices["n"] >= 30)].sort_values("delta_acc_vs_all")
    for _, r in s.head(3).iterrows():
        if r["delta_acc_vs_all"] < -0.01:
            out.append(f"Slice **{r['slice']}** (n={r['n']}) is {abs(r['delta_acc_vs_all']):.1%} below overall "
                       f"accuracy.")
    if len(tags):
        top = tags.head(3)
        out.append("Most frequent error tags: " + ", ".join(f"*{k}* ({v} errors, {v / tags.attrs['n_err']:.0%})"
                                                            for k, v in top.items()) + ".")
    pol = df[~df["correct"] & df["label"].isin(["positive", "negative"]) & df["pred"].isin(["positive", "negative"])]
    n_err = (~df["correct"]).sum()
    if n_err and {"positive", "negative", "neutral"} <= set(df["label"]) | set(df["pred"]):
        out.append(f"{len(pol) / n_err:.0%} of errors are full polarity flips (positive↔negative); the rest "
                   f"involve the neutral class, which is the hardest boundary to annotate consistently.")
    ece = metrics.get("ece")
    if ece is not None:
        verdict = ("well calibrated" if ece < 0.03
                   else "moderately mis-calibrated" if ece < 0.08 else "poorly calibrated")
        out.append(f"ECE = {ece:.3f} → the model is {verdict}; "
                   + ("consider temperature scaling before using confidences as probabilities." if ece >= 0.03
                      else "confidence scores can be used for thresholding / abstention."))
    return out

def run_error_analysis(cfg: Config, predictions_path: str | Path, model_dir: str | Path | None = None,
                       baseline: dict | None = None) -> Path:
    out = Path(cfg.analysis.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(predictions_path)
    labels = cfg.labels

    tokenizer = None
    if model_dir is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    df = add_features(df, tokenizer, cfg.model.max_length)
    y, p = df["label_id"].to_numpy(), df["pred_id"].to_numpy()
    probs = df[[f"p_{lbl}" for lbl in labels]].to_numpy()
    metrics = classification_metrics(y, p, labels, df["language"].tolist())
    metrics["ece"] = expected_calibration_error(probs, y, cfg.analysis.calibration_bins)
    by_source = classification_metrics(y, p, labels, df["source"].tolist())["by_group"]
    ece_by_lang = {lang: expected_calibration_error(probs[m], y[m], cfg.analysis.calibration_bins)
                   for lang in sorted(df["language"].unique()) for m in [df["language"].to_numpy() == lang]}
    baseline = baseline or {"accuracy": float("nan"), "macro_f1": float("nan")}
    slices = slice_table(df, labels)
    pairs = confusion_pairs(df)
    errors = tag_errors(df[~df["correct"]])
    tag_counts = errors["auto_tags"].str.split("; ").explode().value_counts()
    tag_counts.attrs["n_err"] = max(len(errors), 1)
    tag_by_lang = (errors.assign(tag=errors["auto_tags"].str.split("; ")).explode("tag")
                   .pivot_table(index="tag", columns="language", values="text", aggfunc="count", fill_value=0))

    plot_confusions(df, labels, out / "figures" / "confusion_by_language.png")
    plot_calibration(df, cfg.analysis.calibration_bins, out / "figures" / "calibration.png")
    plot_slices(slices, metrics["accuracy"], out / "figures" / "slice_accuracy.png")

    k = cfg.analysis.top_k_examples
    confident_errors = errors.sort_values("confidence", ascending=False).head(k)
    sample = errors.sample(frac=1.0, random_state=cfg.seed).groupby("language").head(40)
    annot_cols = ["source", "language", "text", "label", "pred", "confidence", "auto_tags"]
    annot = sample[annot_cols].assign(manual_category="", gold_label_correct="", notes="")
    annot.to_csv(out / "manual_annotation_sheet.csv", index=False, encoding="utf-8-sig")
    errors[annot_cols].to_csv(out / "all_errors.csv", index=False, encoding="utf-8-sig")
    slices.to_csv(out / "slices.csv", index=False)
    pairs.to_csv(out / "confusion_pairs.csv", index=False)
    write_json({"metrics": metrics, "ece_by_language": ece_by_lang, "by_source": by_source, "baseline": baseline,
                "error_tags": tag_counts.to_dict()}, out / "summary.json")

    lang_rows = pd.DataFrame([
        {"language": LANG_NAMES.get(g, g), "n": v["n"], "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
         "ECE": ece_by_lang[g], "labels present": ", ".join(v["labels_present"])}
        for g, v in metrics["by_group"].items()])
    src_rows = pd.DataFrame([{"source": g, **{k2: v[k2] for k2 in ("n", "accuracy", "macro_f1")}}
                             for g, v in by_source.items()])
    pc = metrics["per_class"]
    class_rows = pd.DataFrame([{"class": c, "precision": pc[c]["precision"], "recall": pc[c]["recall"],
                                "f1": pc[c]["f1-score"], "support": int(pc[c]["support"])} for c in labels])

    md = [
        "# Error Analysis — Multilingual (English / Urdu / Roman Urdu) Sentiment with XLM-R",
        "",
        f"Evaluated on **{len(df):,}** held-out examples; **{len(errors):,}** errors "
        f"({len(errors) / max(len(df), 1):.1%}).",
        "",
        "## 1. Key findings",
        "",
        *[f"- {f}" for f in _findings(df, slices, tag_counts, metrics, baseline)],
        "",
        "## 2. Headline metrics",
        "",
        f"| model | accuracy | macro-F1 | weighted-F1 | ECE |\n|---|---|---|---|---|\n"
        f"| majority baseline | {baseline['accuracy']:.3f} | {baseline['macro_f1']:.3f} | – | – |\n"
        f"| fine-tuned XLM-R | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} | "
        f"{metrics['weighted_f1']:.3f} | {metrics['ece']:.3f} |",
        "",
        "### Per class", "", _md_table(class_rows), "",
        "### Per language", "", _md_table(lang_rows), "",
        "> Note: Urdu-script sources are binary (positive/negative), so macro-F1 for `ur` is computed over the "
        "labels actually present; predictions of *neutral* on Urdu-script text still count as errors.", "",
        "### Per source dataset", "", _md_table(src_rows), "",
        "## 3. Confusion matrices", "", "![confusion](figures/confusion_by_language.png)", "",
        "### Most frequent confusions", "",
        _md_table(pairs.groupby("language").head(4).rename(columns={"label": "true", "pred": "predicted"})), "",
        "## 4. Slice analysis", "",
        "Accuracy on linguistically-defined subsets. Δ is the difference from overall accuracy; "
        "slices with n < 10 are omitted.", "",
        _md_table(slices), "", "![slices](figures/slice_accuracy.png)", "",
        "## 5. Error taxonomy (automatic tags)", "",
        "Each error is tagged with every heuristic category it matches (an error can have several tags).", "",
        _md_table(tag_counts.rename("errors").reset_index().rename(columns={"index": "tag", "auto_tags": "tag"})
                  .assign(share=lambda d: d["errors"] / tag_counts.attrs["n_err"])), "",
        "### Tags by language", "", _md_table(tag_by_lang.reset_index()), "",
        "## 6. Calibration", "", "![calibration](figures/calibration.png)", "",
        "## 7. Examples", "",
    ]
    for (lang, true, pred), grp in errors.groupby(["language", "label", "pred"]):
        if len(grp) < 3:
            continue
        md += [f"**{LANG_NAMES.get(lang, lang)}: true `{true}` → predicted `{pred}`** ({len(grp)} errors)", ""]
        for _, r in grp.sort_values("confidence", ascending=False).head(3).iterrows():
            md.append(f"- ({r['confidence']:.2f}) {_clip(r['text'])}  — _{r['auto_tags']}_")
        md.append("")
    md += [
        "### Most confident errors (label-noise candidates)", "",
        "High-confidence errors are disproportionately annotation mistakes. Review these before blaming the model.",
        "", _md_table(confident_errors[["language", "label", "pred", "confidence", "text"]]
                      .assign(text=lambda d: d["text"].map(_clip))), "",
        "## 8. Manual analysis protocol", "",
        "`manual_annotation_sheet.csv` contains a stratified sample of up to 40 errors per language. "
        "For each row, fill:",
        "", "- `gold_label_correct` — yes / no / ambiguous (is the dataset label itself right?)",
        "- `manual_category` — one of: negation, sarcasm/irony, mixed sentiment, domain/world knowledge, "
        "neutral-vs-weak-polarity, transliteration/spelling variant, code-switching, truncation, label noise, other",
        "- `notes` — free text", "",
        "Then report the category distribution per language; this complements the automatic tags above, "
        "which are high-recall heuristics rather than ground truth.", "",
        "## 9. Artifacts", "",
        "- `all_errors.csv` — every misclassified example with auto tags",
        "- `slices.csv`, `confusion_pairs.csv`, `summary.json` — machine-readable results",
        "- `figures/` — PNG charts used above", "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    log.info("error analysis written to %s", report)
    return report
