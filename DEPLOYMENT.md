# Deploying the Streamlit app for free (Streamlit Community Cloud)

```
 Colab (GPU) ── train ──► Hugging Face Hub (stores the ~1 GB model, free)
                                   │  downloaded on first start
 GitHub repo (code + report) ──► Streamlit Community Cloud ──► https://<your-app>.streamlit.app
```

**Why the model goes on Hugging Face Hub:** GitHub rejects files larger than 100 MB, and the trained
XLM-R model is about 1.1 GB. Streamlit Cloud downloads the model from the Hub the first time the app
starts and then keeps it cached.

> The app works **before** you train anything. When no model is configured, it uses the public
> `cardiffnlp/twitter-xlm-roberta-base-sentiment` (also XLM-R, same 3 labels) and shows a notice in
> the sidebar. You can deploy right away and switch to your own model later.

---

## Step 1: Run it locally first (VS Code)

```powershell
.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```
Or use **Run and Debug → "0. Streamlit app"**. It opens at http://localhost:8501.

## Step 2: Train and publish your model (Google Colab, free GPU)

1. Create a free account at https://huggingface.co. Go to *Settings → Access Tokens → New token*
   and choose the **Write** type.
2. In Colab, choose *Runtime → Change runtime type → T4 GPU*, then run:
   ```python
   !git clone https://github.com/abdulahadwork-code/Multilingual-sentiment-toxicity-detection.git project
   %cd project
   !pip install -q -e ".[serve]"
   !xlmr-sentiment pipeline                      # prepare data, train, evaluate, error analysis (~45-60 min)
   !hf auth login                                # paste your WRITE token
   !xlmr-sentiment push-to-hub <hf-username>/xlmr-urdu-english-sentiment
   !zip -r report.zip reports/error_analysis     # download this zip
   ```
3. Unzip `report.zip` into your local project so `reports/error_analysis/report.md` exists. The
   app's **Error analysis** tab displays it.

## Step 3: Push the code to GitHub

```powershell
git init
git add .
git commit -m "Multilingual sentiment app"
git branch -M main
git remote add origin https://github.com/abdulahadwork-code/Multilingual-sentiment-toxicity-detection.git
git push -u origin main
```
`.gitignore` already excludes the virtual environment, data, model weights and secrets.

## Step 4: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Repository: `abdulahadwork-code/Multilingual-sentiment-toxicity-detection`, Branch: `main`, Main file path: `streamlit_app.py`.
4. Open **Advanced settings**:
   - **Python version: 3.12**
   - **Secrets** (TOML):
     ```toml
     MODEL_REPO = "<hf-username>/xlmr-urdu-english-sentiment"
     # HF_TOKEN = "hf_..."   # only needed if you made the Hub repo private
     ```
5. Click **Deploy**. The first build installs PyTorch and downloads the model, which takes about
   3–6 minutes. Later visits are fast.

To change the model later, go to *App → Settings → Secrets*, edit `MODEL_REPO` and reboot the app.

---

## How the app fits in the free tier

| Concern | What the code does |
|---|---|
| Memory (free apps get roughly 0.7–2.7 GB) | CPU-only torch (~1.2 GB RAM in fp32); only PyTorch weights downloaded; the model loads once via `st.cache_resource` |
| Download size | Only the PyTorch weights and tokenizer files are fetched, not the TF/ONNX copies |
| Slow CPU | Batches are sorted by length; at most 2 threads; batch mode is capped at 2,000 rows |
| Abuse / huge inputs | Text length and row limits; user text is HTML-escaped before display |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "This app has gone over its resource limits" | Reboot from *Manage app*. If it keeps happening, reduce `MAX_BATCH_ROWS` in `streamlit_app.py`. |
| Build takes forever / installs CUDA | Make sure `requirements.txt` still has the `--extra-index-url .../whl/cpu` line. |
| `Could not load the model` | Check that the `MODEL_REPO` spelling matches the Hub repo, and add `HF_TOKEN` if the repo is private. |
| App is "asleep" | Free apps sleep after 12 h without visitors. Click *Yes, get this app back up*. |
| Error-analysis tab is empty | Commit `reports/error_analysis/` (the report and its `figures/`). |

## Alternative free host: Hugging Face Spaces

Spaces gives 16 GB RAM on the free CPU tier, which is more headroom than Streamlit Cloud. Create a
Space, add `streamlit_app.py`, `requirements.txt` and `src/` to it, and set `MODEL_REPO` under
*Settings → Variables*.
