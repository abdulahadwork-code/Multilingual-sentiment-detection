# Multilingual Sentiment Analysis: English / Urdu / Roman Urdu with fine-tuned XLM-RoBERTa

Three-class sentiment classification (**negative / neutral / positive**) for English, Urdu written in
Arabic script (اردو), and Roman Urdu ("yeh movie bohat achi thi"). The project uses a fine-tuned
**XLM-RoBERTa** (`xlm-roberta-base`, or `-large`) and includes a full **error analysis** module. A
production REST API serves the trained model.

```
raw HF datasets ─► prepare-data ─► train (XLM-R) ─► evaluate ─► error analysis report
                   (normalise,      (class-weighted   (per-language   (slices, taxonomy,
                    harmonise,       CE, early stop,   metrics, ECE)   calibration, examples)
                    de-leak)         best-by-macro-F1)
                                          │
                                          ▼
                              FastAPI service (Docker, /v1/predict)
```

## 1. Problem and motivation

Urdu has more than 230 million speakers, but it is a low-resource language for NLP. Online, Pakistani
users write it in two scripts: Nastaliq/Arabic script and an informal romanised form with no fixed
spelling (`nahi / nhi / nai`). Many messages also switch to English in the middle of a sentence. A
monolingual English model can't handle this. XLM-R was pre-trained on 2.5 TB of CommonCrawl text in
100 languages, Urdu included, so it gives a shared representation for all three varieties.

## 2. Data

| Source (Hugging Face) | Variety | Labels | Used rows |
|---|---|---|---|
| `cardiffnlp/tweet_eval` (sentiment) | English tweets | neg / neu / pos | 20k train + official val/test |
| `community-datasets/roman_urdu` | Roman Urdu comments | neg / neu / pos | ~20k, split 80/10/10 |
| `sepidmnorozy/Urdu_sentiment` | Urdu news/tweets | neg / pos | ~1k |
| `mirfan899/imdb_urdu_reviews` | Urdu movie reviews (translated IMDB) | neg / pos | 12k train / 1.5k val / 1.5k test |

`prepare-data` performs these steps:
* **Harmonises labels** into one label space via each source's `label_map` in `configs/default.yaml`.
* **Normalises text**: it unifies Arabic and Urdu code points (ي→ی, ك→ک, ه→ہ), strips diacritics and
  tatweel, maps Urdu digits to ASCII, replaces URLs and mentions with placeholders, and collapses
  repeated characters. The same function runs at inference time, so preprocessing can't drift
  between training and serving.
* **Splits**: it uses the official splits when they exist. Otherwise it creates stratified splits
  with a fixed seed.
* **De-duplicates and de-leaks**: it drops exact duplicates and texts that have conflicting labels.
  It also removes from train/validation any text that also appears in test.
* Writes `data/processed/{train,validation,test}.parquet` and `stats.json`.

To add your own data (for example a toxicity dataset), add a `sources:` entry with `local_path:` set
to a CSV, JSONL or Parquet file, or to a directory containing `train.* / validation.* / test.*`.

## 3. Model and training

* `AutoModelForSequenceClassification` on top of XLM-R. Settings: max 160 tokens, AdamW, lr 2e-5,
  linear schedule with 6% warm-up, 3 epochs, batch 32.
* **Inverse-frequency class-weighted cross-entropy.** Two Urdu-script sources have no *neutral*
  class, so the neutral class is under-represented.
* Evaluation every 500 steps. Early stopping (patience 2) keeps the checkpoint with the best
  **validation macro-F1**. Per-language macro-F1 is logged as well.
* fp16 turns on automatically when a GPU is present. Training uses length-grouped batching.
* `training_meta.json` records the versions, data size, config and metrics alongside the model.

Expected cost: about 40–60 min for `xlm-roberta-base` on a free Colab T4. Use `xlm-roberta-large`
(`--set model.name=FacebookAI/xlm-roberta-large --set training.train_batch_size=16
--set training.learning_rate=1e-5`) on a GPU with 16 GB or more for higher accuracy.

## 4. Error analysis (the `analyze` command)

The command writes `reports/error_analysis/report.md` along with CSVs and figures:

1. **Headline metrics** compared with a majority-class baseline, broken down per class, per language
   and per source dataset.
2. **Confusion matrices per language** (row-normalised) and the most frequent confusion pairs.
3. **Slice analysis** (accuracy and macro-F1 per slice): sentence length, negation (`not / nahi /
   نہیں`), contrast markers (`but / lekin / magar / لیکن`), sarcasm cues, intensifiers, emoji,
   questions, code-mixing, inputs that get truncated, and cases where the detected language differs
   from the source language.
4. **Error taxonomy**: each error gets automatic tags such as negation, mixed sentiment, possible
   sarcasm, neutral boundary, polarity flip, very short, truncated, code-mixed, or confident error
   (possible label noise). Counts are given per language.
5. **Calibration**: a reliability diagram and Expected Calibration Error (ECE) per language.
6. **Qualitative examples** for each confusion type, plus the **most confident errors**. These are
   strong candidates for annotation mistakes.
7. **A manual annotation sheet** (`manual_annotation_sheet.csv`): a stratified sample of 40 errors
   per language with empty `manual_category / gold_label_correct / notes` columns. Fill it in by hand
   to write the qualitative part of the report. The protocol is described in the report.

The *Key findings* section at the top of the report is generated from the numbers. Read the actual
examples before you write your conclusions.

## 5. Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel on a GPU box
pip install -e ".[serve,dev]"
```

Pipeline smoke test (about 2 min on CPU, tiny model and synthetic data, verifies everything runs):

```bash
xlmr-sentiment pipeline -c configs/smoke.yaml
```

Full run (needs a GPU):

```bash
xlmr-sentiment prepare-data            # downloads + cleans + splits
xlmr-sentiment train                   # fine-tunes XLM-R -> artifacts/xlmr-sentiment/best
xlmr-sentiment evaluate                # test-set metrics -> artifacts/xlmr-sentiment/eval
xlmr-sentiment analyze                 # error analysis -> reports/error_analysis/report.md
# or all four:  xlmr-sentiment pipeline
```

To override any config value from the command line, use: `--set training.epochs=5 --set model.max_length=256`

### Google Colab

```bash
!git clone https://github.com/abdulahadwork-code/Multilingual-sentiment-toxicity-detection.git project && cd project && pip install -e ".[serve]"
!xlmr-sentiment pipeline
!zip -r model.zip artifacts/xlmr-sentiment/best reports/error_analysis
```

Download `model.zip` and unzip it into the project on your machine to serve the model locally.

## 6. Inference and serving

```bash
xlmr-sentiment predict "The food was amazing" "یہ فلم بہت بری تھی" "service theek thi"
xlmr-sentiment serve --port 8000     # demo UI at http://localhost:8000, docs at /docs
```

```bash
curl -s localhost:8000/v1/predict -H "Content-Type: application/json" \
  -d '{"texts": ["khana bohat acha tha", "I did not like it at all"]}'
```

```json
{"model_version": "1.0.0", "predictions": [
  {"text": "khana bohat acha tha", "label": "positive", "confidence": 0.97,
   "scores": {"negative": 0.01, "neutral": 0.02, "positive": 0.97}, "language": "ur-Latn", "code_mixed": false}, ...]}
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/predict` | batch classification: up to `MAX_BATCH_SIZE` texts, each up to `MAX_TEXT_CHARS` |
| `GET /v1/model` | labels, base model, validation macro-F1, versions |
| `GET /health` | liveness check (the process is up) |
| `GET /ready` | readiness check (the model is loaded); returns 503 otherwise |
| `GET /metrics` | Prometheus counters and latency histogram |
| `GET /docs` | OpenAPI / Swagger UI |

**Production features:**
* Environment-variable config: `MODEL_DIR`, `DEVICE`, `QUANTIZE_CPU`, `MAX_BATCH_SIZE`,
  `MAX_TEXT_CHARS`, `API_KEY`, `CORS_ORIGINS`, `LOG_FORMAT=json`.
* Optional API-key auth with constant-time comparison.
* Input validation with clear 413/422 errors.
* An `X-Request-ID` header and structured JSON logs on every request.
* Inference runs in a worker thread, so the event loop never blocks. Batches are sorted by length to
  reduce padding.
* Optional dynamic int8 quantisation on CPU (`QUANTIZE_CPU=1`). It is **off by default**: in testing it
  changed XLM-R confidences a lot (0.90 → 0.41 on the same sentence), so check it on your
  validation set before turning it on.
* The model loads once at start-up and a warm-up pass runs. If loading fails, `/health` stays up and
  `/ready` reports the failure.
* The Docker image is multi-stage, runs as a non-root user, uses CPU-only torch, runs offline
  (`HF_HUB_OFFLINE=1`) and has a `HEALTHCHECK`.

### Docker

```bash
docker build -t xlmr-sentiment .        # copies artifacts/xlmr-sentiment/best into the image
docker compose up -d
```

### Streamlit web app (free deployment)

```bash
streamlit run streamlit_app.py      # http://localhost:8501
```
The app has four tabs: single or multi-line analysis with example buttons, batch CSV upload with a
downloadable result, the error-analysis report, and an About page. It loads your model from the
Hugging Face Hub (`MODEL_REPO` secret). Until you set one, it falls back to the public
`cardiffnlp/twitter-xlm-roberta-base-sentiment`. Step-by-step free deployment on Streamlit
Community Cloud is in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

## 7. Testing

```bash
pytest -m "not slow"     # unit + API tests, no network, ~10 s
pytest                   # + end-to-end train/eval/analysis with a tiny XLM-R (downloads ~1 MB once)
```

CI (`.github/workflows/ci.yml`) runs ruff and the full test suite on every push.

## 8. Project layout

```
configs/            default.yaml (real experiment), smoke.yaml (CPU smoke test)
src/xlmr_sentiment/
  config.py         typed (pydantic) config + --set overrides
  text.py           Urdu/Roman-Urdu normalisation, script/language ID, negation cues
  data.py           load -> harmonise -> clean -> split -> de-leak -> parquet
  train.py          HF Trainer + class-weighted loss + early stopping
  metrics.py        macro-F1, per-group metrics, ECE
  evaluate.py       predictions + metrics on a split, majority baseline
  error_analysis.py slices, taxonomy, calibration, figures, markdown report
  inference.py      thread-safe batched predictor (+ CPU quantisation)
  api/app.py        FastAPI service, api/static/index.html demo UI
  cli.py            `xlmr-sentiment` command
tests/              unit, API and end-to-end tests (+ synthetic fixtures)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## 9. Limitations and ethics

* **Domain shift**: the Urdu-script sentiment data is mostly translated IMDB reviews plus news.
  Expect lower accuracy on informal Urdu-script social media text, and use the error analysis to
  measure it.
* **Neutral class**: the Urdu-script sources have no neutral label, so neutral predictions on Urdu
  script come only from cross-lingual transfer.
* **Roman Urdu spelling variation**: unseen spellings are handled through XLM-R subwords, but
  transliteration noise remains a major error source.
* **Sarcasm** and **culturally specific idioms** are hard for every model. Don't use the model for
  high-stakes moderation decisions without human review.
* The language detector is a simple heuristic based on character scripts and common Roman-Urdu
  words. It's meant for analysis and response metadata, not as a gate.

## 10. Extending to toxicity detection

Toxicity uses the same pipeline. Set `labels: [non_toxic, toxic]` and point the `sources` at toxicity
datasets, for example an English Jigsaw sample plus an Urdu or Roman-Urdu abusive-language corpus. No
code changes are needed. The error-analysis slices (code-mixing, negation, short texts, confident
errors) apply directly.
