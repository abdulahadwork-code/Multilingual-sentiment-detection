from __future__ import annotations

import html
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from xlmr_sentiment.inference import SentimentPredictor  

AUTHOR_NAME = "Abdul Ahad"
GITHUB_URL = "https://github.com/abdulahadwork-code/Multilingual-sentiment-toxicity-detection"
LINKEDIN_URL = ""

LOCAL_MODEL = ROOT / "artifacts" / "xlmr-sentiment" / "best"
FALLBACK_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
REPORT_DIR = ROOT / "reports" / "error_analysis"
MAX_CHARS = 2000
MAX_BATCH_ROWS = 2000

SENTIMENT = {
    "negative": {"color": "#d03b3b", "icon": "▼", "name": "Negative"},
    "neutral": {"color": "#8a8983", "icon": "●", "name": "Neutral"},
    "positive": {"color": "#0ca30c", "icon": "▲", "name": "Positive"},
}
LANG_NAMES = {"en": "English", "ur": "Urdu", "ur-Latn": "Roman Urdu", "unknown": "Unknown"}
EXAMPLES = {
    "English · positive": "The delivery was quick and the product quality is excellent!",
    "English · negative": "I waited two hours and nobody even answered my call. Terrible.",
    "Urdu · positive": "یہ فلم واقعی شاندار تھی، اداکاری بہت زبردست تھی",
    "Urdu · negative": "یہ فلم بالکل بکواس تھی، وقت ضائع ہوا",
    "Roman Urdu · neutral": "khana theek tha, kuch khaas nahi",
    "Code-mixed": "Food was great lekin service bohat slow thi",
}

st.set_page_config(page_title="Urdu–English Sentiment Analyzer", page_icon="💬", layout="centered",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');
:root {
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #6f6e69; --line: #e4e3df; --surface: #ffffff;
  --page: #f9f9f7; --accent: #2a78d6; --accent-soft: #eaf2fc;
}
html, body, .stMarkdown, .stMarkdown p, label, input, textarea, [data-testid="stWidgetLabel"] p,
.stButton button p, [data-baseweb="tab"] p { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: var(--page); }
.block-container { max-width: 920px; padding: 2.5rem 1rem 3rem; }

.hero { padding: 8px 0 18px; }
.eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.hero h1 { font-size: clamp(1.7rem, 4.2vw, 2.5rem); line-height: 1.15; font-weight: 700; color: var(--ink);
           margin: .35rem 0 .6rem; padding: 0; letter-spacing: -.02em; }
.lede { font-size: 1.02rem; color: var(--ink-2); max-width: 680px; margin: 0 0 14px; line-height: 1.6; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.chip { font-size: .8rem; color: var(--ink-2); background: var(--surface); border: 1px solid var(--line);
        border-radius: 999px; padding: 3px 11px; }
.links { font-size: .88rem; color: var(--ink-3); }
.links a { color: var(--accent); text-decoration: none; font-weight: 500; }
.links a:hover { text-decoration: underline; }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px;
        margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,.03); }
.quote { font-size: 1.02rem; color: var(--ink); margin-bottom: 14px; unicode-bidi: plaintext; line-height: 1.6; }
.quote.ur { font-family: 'Noto Nastaliq Urdu', serif; direction: rtl; text-align: right; line-height: 2.3; font-size: 1.1rem; }
.verdict { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.badge { width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center; color: #fff; font-size: 1rem; flex: none; }
.vlabel { font-size: 1.25rem; font-weight: 700; color: var(--ink); line-height: 1.1; }
.vsub { font-size: .85rem; color: var(--ink-2); }
.meta { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }
.bars { display: grid; grid-template-columns: 78px 1fr 52px; gap: 8px 10px; align-items: center; font-size: .85rem; }
.bars .name { color: var(--ink-2); }
.bars .pct { color: var(--ink); text-align: right; font-variant-numeric: tabular-nums; }
.track { height: 8px; background: #f0efec; border-radius: 4px; overflow: hidden; }
.fill { height: 100%; border-radius: 4px; }
.compact .verdict { margin-bottom: 0; }
.compact .quote { margin-bottom: 10px; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 8px 0 18px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.tile .k { font-size: .78rem; color: var(--ink-2); }
.tile .v { font-size: 1.6rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile .s { font-size: .75rem; color: var(--ink-3); }
.note { background: var(--accent-soft); border-radius: 12px; padding: 14px 16px; color: var(--ink); font-size: .92rem; line-height: 1.55; }
.steps { counter-reset: s; display: grid; gap: 10px; }
.step { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px 12px 52px; position: relative; color: var(--ink-2); font-size: .93rem; line-height: 1.55; }
.step b { color: var(--ink); }
.step::before { counter-increment: s; content: counter(s); position: absolute; left: 16px; top: 12px; width: 24px; height: 24px; border-radius: 50%;
                background: var(--accent-soft); color: var(--accent); font-weight: 700; font-size: .8rem; display: grid; place-items: center; }
.foot { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line); font-size: .82rem; color: var(--ink-3);
        display: flex; flex-wrap: wrap; gap: 6px 14px; justify-content: space-between; }
.foot a { color: var(--ink-2); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; font-weight: 500; }
@media (max-width: 640px) {
  .block-container { padding-top: 1.5rem; }
  .meta { margin-left: 0; width: 100%; }
  .bars { grid-template-columns: 70px 1fr 46px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  
        value = None
    return value or os.getenv(name) or None
def model_source() -> tuple[str, bool]:
    """Return (source, is_baseline)."""
    configured = _secret("MODEL_REPO")
    if configured:
        return configured, False
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL), False
    return FALLBACK_MODEL, True
def model_label(source: str) -> str:
    return "local fine-tuned model" if Path(source).exists() else source
def model_url(source: str) -> str | None:
    is_hub_id = source.count("/") == 1 and not Path(source).exists()
    return f"https://huggingface.co/{source}" if is_hub_id else None

@st.cache_resource(show_spinner="Loading the model. The first start downloads about 1 GB…")
def load_predictor(source: str) -> SentimentPredictor:
    import torch

    token = _secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))  
    return SentimentPredictor(source, device="cpu", quantize_cpu=False, batch_size=32, max_chars=MAX_CHARS,
                              max_length=160)
@st.cache_data
def load_summary() -> dict | None:
    path = REPORT_DIR / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
def esc(s: str) -> str:
    return html.escape(str(s))
def chip(text: str) -> str:
    return f'<span class="chip">{esc(text)}</span>'
def result_card(text: str, pred, compact: bool = False) -> str:
    s = SENTIMENT.get(pred.label, {"color": "#2a78d6", "icon": "•", "name": pred.label.title()})
    lang = LANG_NAMES.get(pred.language, pred.language)
    meta = chip(lang) + (chip("code-mixed") if pred.code_mixed else "")
    rows = "".join(
        f'<span class="name">{esc(SENTIMENT.get(lbl, {}).get("name", lbl))}</span>'
        f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%;'
        f'background:{SENTIMENT.get(lbl, {}).get("color", "#2a78d6")}"></div></div>'
        f'<span class="pct">{p:.1%}</span>'
        for lbl, p in pred.scores.items()
    )
    ur = " ur" if pred.language == "ur" else ""
    return f"""
<div class="card{' compact' if compact else ''}">
  <div class="quote{ur}" dir="auto">{esc(text)}</div>
  <div class="verdict">
    <div class="badge" style="background:{s['color']}" aria-hidden="true">{s['icon']}</div>
    <div><div class="vlabel">{esc(s['name'])}</div><div class="vsub">{pred.confidence:.1%} confidence</div></div>
    <div class="meta">{meta}</div>
  </div>
  {'' if compact else f'<div class="bars">{rows}</div>'}
</div>"""

def tiles(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
                    f'<div class="s">{esc(s)}</div></div>' for k, v, s in items)
    return f'<div class="tiles">{cells}</div>'

def render_report(path: Path) -> None:
    """Render the error-analysis markdown, swapping relative image links for st.image calls."""
    buf: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"!\[(.*?)\]\((.+?)\)", line.strip())
        if m:
            st.markdown("\n".join(buf))
            buf = []
            img = path.parent / m.group(2)
            if img.exists():
                st.image(str(img), caption=m.group(1), width="stretch")
        else:
            buf.append(line)
    st.markdown("\n".join(buf))

def _use_example() -> None:
    pick = st.session_state.get("example_pick")
    if pick:
        st.session_state["input_text"] = EXAMPLES[pick]
        st.session_state.pop("results", None)

source, is_baseline = model_source()
github = _secret("GITHUB_URL") or GITHUB_URL
linkedin = _secret("LINKEDIN_URL") or LINKEDIN_URL
author = _secret("AUTHOR_NAME") or AUTHOR_NAME
links = [f"Built by <b>{esc(author)}</b>"]
if github:
    links.append(f'<a href="{esc(github)}" target="_blank">Source code</a>')
if model_url(source) and not is_baseline:
    links.append(f'<a href="{esc(model_url(source))}" target="_blank">Model card</a>')
if linkedin:
    links.append(f'<a href="{esc(linkedin)}" target="_blank">LinkedIn</a>')

st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">Deep learning · NLP</div>
  <h1>Urdu–English Sentiment Analyzer</h1>
  <p class="lede">Detects whether a text is negative, neutral or positive in <b>English</b>,
  <b>Urdu</b> (<span dir="rtl">اردو</span>) and <b>Roman Urdu</b>, including code-mixed sentences
  such as <i>“food was great lekin service slow thi”</i>. Powered by a fine-tuned XLM-RoBERTa transformer.</p>
  <div class="chips">{chip("XLM-RoBERTa")}{chip("3 languages · 2 scripts")}{chip("negative · neutral · positive")}{chip("Error analysis included")}</div>
  <div class="links">{" &nbsp;·&nbsp; ".join(links)}</div>
</div>""",
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor(source)
except Exception as e:  # show a friendly error instead of a stack trace
    st.error(f"The model could not be loaded from `{source}`. Details: {e}")
    st.stop()

tab_try, tab_batch, tab_eval, tab_about = st.tabs(["Analyze", "Batch analysis", "Model & evaluation", "How it works"])

with tab_try:
    st.pills("Try an example", list(EXAMPLES), key="example_pick", on_change=_use_example)
    text = st.text_area("Your text", key="input_text", height=120, max_chars=MAX_CHARS * 5,
                        placeholder="Type or paste English, اردو or Roman Urdu. Put each text on its own line to "
                                    "analyse several at once.")
    if st.button("Analyze sentiment", type="primary", width="stretch"):
        lines = [t.strip() for t in text.splitlines() if t.strip()][:50]
        if not lines:
            st.info("Type some text or pick an example first.")
        else:
            with st.spinner("Analyzing…"):
                st.session_state["results"] = list(zip(lines, predictor.predict(lines), strict=True))
    results = st.session_state.get("results")
    if results:
        compact = len(results) > 3
        st.markdown("".join(result_card(t, p, compact) for t, p in results), unsafe_allow_html=True)

with tab_batch:
    st.markdown(f"Upload a CSV file with a column of text (up to {MAX_BATCH_ROWS:,} rows). You get a "
                "prediction for every row, a summary by language, and a downloadable results file.")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except UnicodeDecodeError:
            up.seek(0)
            df = pd.read_csv(up, encoding="latin-1")
        text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        default = next((c for c in text_cols if str(c).lower() in ("text", "sentence", "review", "comment")),
                       text_cols[0])
        col = st.selectbox("Which column contains the text?", text_cols, index=text_cols.index(default))
        if len(df) > MAX_BATCH_ROWS:
            st.caption(f"Only the first {MAX_BATCH_ROWS:,} of {len(df):,} rows will be analysed.")
            df = df.head(MAX_BATCH_ROWS)
        if st.button("Analyze file", type="primary"):
            texts = df[col].fillna("").astype(str).tolist()
            bar = st.progress(0.0, text="Analyzing…")
            preds = []
            for start in range(0, len(texts), 64):
                chunk = [t if t.strip() else " " for t in texts[start:start + 64]]
                preds.extend(predictor.predict(chunk))
                done = min(start + 64, len(texts))
                bar.progress(done / len(texts), text=f"Analyzed {done:,} / {len(texts):,}")
            bar.empty()
            out = df.copy()
            out["sentiment"] = [p.label for p in preds]
            out["confidence"] = [p.confidence for p in preds]
            out["language"] = [LANG_NAMES.get(p.language, p.language) for p in preds]
            for lbl in predictor.labels:
                out[f"p_{lbl}"] = [p.scores[lbl] for p in preds]
            st.session_state["batch_out"] = out
    out = st.session_state.get("batch_out")
    if out is not None:
        n = len(out)
        share = out["sentiment"].value_counts(normalize=True)
        st.markdown(tiles([("Rows analysed", f"{n:,}", "")] + [
            (SENTIMENT[lbl]["name"], f"{share.get(lbl, 0):.0%}", f"{int(share.get(lbl, 0) * n):,} rows")
            for lbl in predictor.labels if lbl in SENTIMENT]), unsafe_allow_html=True)
        counts = out.groupby(["language", "sentiment"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=[lbl for lbl in predictor.labels if lbl in counts.columns])
        st.markdown("**Sentiment by language**")
        st.bar_chart(counts, color=[SENTIMENT.get(c, {}).get("color", "#2a78d6") for c in counts.columns],
                     horizontal=True, height=60 + 48 * len(counts))
        st.dataframe(out, width="stretch", height=320, hide_index=True)
        csv = io.StringIO()
        out.to_csv(csv, index=False)
        st.download_button("Download results (CSV)", csv.getvalue().encode("utf-8-sig"),
                           file_name="sentiment_predictions.csv", mime="text/csv")
with tab_eval:
    summary = load_summary()
    meta = predictor.meta or {}
    if summary and not is_baseline:  
        m = summary["metrics"]
        base = summary.get("baseline", {})
        st.markdown("#### Test-set results")
        st.markdown(tiles([
            ("Macro-F1", f"{m['macro_f1']:.3f}",
             f"majority baseline {base['macro_f1']:.3f}"
             if isinstance(base.get("macro_f1"), float) and base["macro_f1"] == base["macro_f1"] else ""),
            ("Accuracy", f"{m['accuracy']:.1%}", ""),
            ("Calibration error (ECE)", f"{m.get('ece', float('nan')):.3f}", "lower is better"),
            ("Test examples", f"{m['n']:,}", ""),
        ]), unsafe_allow_html=True)
        by_lang = pd.DataFrame([
            {"Language": LANG_NAMES.get(g, g), "Examples": v["n"], "Accuracy": f"{v['accuracy']:.1%}",
             "Macro-F1": f"{v['macro_f1']:.3f}",
             "ECE": f"{summary.get('ece_by_language', {}).get(g, float('nan')):.3f}"}
            for g, v in m["by_group"].items()])
        st.markdown("**By language**")
        st.dataframe(by_lang, hide_index=True, width="stretch")
        for fig, title in (("confusion_by_language.png", "Confusion matrices"),
                           ("slice_accuracy.png", "Where the model struggles"),
                           ("calibration.png", "Calibration")):
            if (REPORT_DIR / "figures" / fig).exists():
                st.markdown(f"**{title}**")
                st.image(str(REPORT_DIR / "figures" / fig), width="stretch")
        if (REPORT_DIR / "report.md").exists():
            with st.expander("Read the full error-analysis report"):
                render_report(REPORT_DIR / "report.md")
    elif meta.get("validation_metrics"):
        vm = meta["validation_metrics"]
        st.markdown(tiles([
            ("Validation macro-F1", f"{vm.get('eval_macro_f1', float('nan')):.3f}", ""),
            ("Validation accuracy", f"{vm.get('eval_accuracy', float('nan')):.1%}", ""),
            ("Training examples", f"{meta.get('train_rows', 0):,}", ""),
        ]), unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="note">This demo is currently running the public XLM-R sentiment baseline '
            f'(<code>{esc(FALLBACK_MODEL)}</code>). Test-set metrics, confusion matrices and the full error '
            f'analysis for the fine-tuned English/Urdu/Roman-Urdu model will appear here once it is '
            f'published.</div>' if is_baseline else
            '<div class="note">No evaluation results were found for this model. Run '
            '<code>xlmr-sentiment analyze</code> and commit <code>reports/error_analysis/</code>.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("#### Training setup")
    st.table(pd.DataFrame([
        ("Base model", meta.get("base_model", "FacebookAI/xlm-roberta-base")),
        ("Training data", "≈48k examples: English tweets (TweetEval), Roman-Urdu comments, Urdu news and reviews"),
        ("Objective", "Class-weighted cross-entropy, 3 classes"),
        ("Optimisation", "AdamW · lr 2e-5 · linear schedule, 6% warm-up · 3 epochs · early stopping"),
        ("Model selection", "Best validation macro-F1"),
        ("Max sequence length", f"{predictor.max_length} tokens"),
    ], columns=["Setting", "Value"]).set_index("Setting"))

with tab_about:
    st.markdown(
        """
<div class="steps">
  <div class="step"><b>Normalise the text.</b> Urdu typed on Arabic keyboards uses look-alike characters
  (ي/ی, ك/ک, ه/ہ). These are unified, diacritics are stripped, and URLs and mentions are replaced. Training
  and this app share the exact same function.</div>
  <div class="step"><b>Encode with XLM-RoBERTa.</b> A transformer pre-trained on 2.5 TB of text in 100
  languages turns the sentence into contextual embeddings. One shared vocabulary covers both scripts.</div>
  <div class="step"><b>Classify.</b> A fine-tuned classification head outputs probabilities for negative,
  neutral and positive. The bars show the full distribution, not only the winner.</div>
  <div class="step"><b>Detect the language.</b> A lightweight script and lexicon heuristic tags each input as
  English, Urdu or Roman Urdu and flags code-mixing.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Known limitations")
    st.markdown(
        "- **Sarcasm** and **mixed sentiment** (*“good food lekin bad service”*) are the most common errors.\n"
        "- The Urdu-script training data has **no neutral class**, so neutral predictions on Urdu script rely on "
        "cross-lingual transfer.\n"
        "- Roman Urdu has no standard spelling (*nahi / nhi / nai*), so rare spellings are less reliable.\n"
        "- This is a research demo, so don't use it for high-stakes moderation without human review."
    )
    st.markdown("#### Tech stack")
    st.markdown(
        tiles([("Model", "XLM-R", "PyTorch · Transformers"), ("Serving", "Streamlit", "+ FastAPI REST API"),
               ("Quality", "38 tests", "pytest · ruff · CI")]),
        unsafe_allow_html=True,
    )

foot_model = (f'<a href="{esc(model_url(source))}" target="_blank">{esc(model_label(source))}</a>'
              if model_url(source) else esc(model_label(source)))
st.markdown(
    f'<div class="foot"><span>{" · ".join(links)}</span><span>Model: {foot_model}</span></div>',
    unsafe_allow_html=True,
)
