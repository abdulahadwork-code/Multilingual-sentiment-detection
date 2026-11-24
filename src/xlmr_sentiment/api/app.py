from __future__ import annotations

import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..utils import setup_logging

log = logging.getLogger("xlmr_sentiment.api")
STATIC = Path(__file__).parent / "static"

REQUESTS = Counter("sentiment_requests_total", "HTTP requests", ["endpoint", "status"])
TEXTS = Counter("sentiment_texts_total", "Texts classified", ["label", "language"])
LATENCY = Histogram("sentiment_inference_seconds", "Model inference latency per request",
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))

class PredictorLike(Protocol):
    labels: list[str]
    device: str
    quantized: bool
    meta: dict

    def predict(self, texts: list[str]): ...

class Settings(BaseModel):
    model_dir: str = "artifacts/xlmr-sentiment/best"
    device: str = "auto"
    quantize_cpu: bool = False
    max_batch_size: int = Field(64, ge=1)
    max_text_chars: int = Field(5000, ge=1)
    api_key: str | None = None
    cors_origins: list[str] = []

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ
        return cls(
            model_dir=e.get("MODEL_DIR", cls.model_fields["model_dir"].default),
            device=e.get("DEVICE", "auto"),
            quantize_cpu=e.get("QUANTIZE_CPU", "0").lower() not in ("0", "false", "no"),
            max_batch_size=int(e.get("MAX_BATCH_SIZE", 64)),
            max_text_chars=int(e.get("MAX_TEXT_CHARS", 5000)),
            api_key=e.get("API_KEY") or None,
            cors_origins=[o.strip() for o in e.get("CORS_ORIGINS", "").split(",") if o.strip()],
        )

class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="One or more texts in English, Urdu or Roman Urdu.",
                             examples=[["This phone is amazing!", "یہ فلم بہت بری تھی", "khana theek tha"]])

    @field_validator("texts")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if any(not t or not t.strip() for t in v):
            raise ValueError("texts must not contain empty strings")
        return v

class PredictionOut(BaseModel):
    text: str
    label: str
    confidence: float
    scores: dict[str, float]
    language: str = Field(description="Detected variety: en | ur | ur-Latn | unknown")
    code_mixed: bool

class PredictResponse(BaseModel):
    model_version: str
    predictions: list[PredictionOut]

class ModelInfo(BaseModel):
    api_version: str
    labels: list[str]
    device: str
    quantized: bool
    base_model: str | None = None
    validation_macro_f1: float | None = None
    trained_with: dict | None = None

def create_app(predictor: PredictorLike | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            from ..inference import SentimentPredictor

            try:
                app.state.predictor = SentimentPredictor(
                    settings.model_dir, device=settings.device, quantize_cpu=settings.quantize_cpu,
                    max_chars=settings.max_text_chars)
                app.state.predictor.warmup()
            except Exception:  
                log.exception("failed to load model from %s", settings.model_dir)
        yield

    app = FastAPI(title="Multilingual Sentiment API (EN / UR / Roman-UR)", version=__version__, lifespan=lifespan)
    app.state.predictor = predictor
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
            response = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
        if request.url.path not in ("/health", "/metrics"):
            REQUESTS.labels(request.url.path, str(response.status_code)).inc()
            log.info("%s %s -> %d in %.1fms rid=%s", request.method, request.url.path, response.status_code, ms, rid)
        return response

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def get_predictor() -> PredictorLike:
        p = app.state.predictor
        if p is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return p

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        if app.state.predictor is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", response_model=ModelInfo, tags=["model"], dependencies=[Depends(require_key)])
    def model_info(p: PredictorLike = Depends(get_predictor)):
        meta = p.meta or {}
        return ModelInfo(
            api_version=__version__, labels=p.labels, device=p.device, quantized=p.quantized,
            base_model=meta.get("base_model"),
            validation_macro_f1=(meta.get("validation_metrics") or {}).get("eval_macro_f1"),
            trained_with={k: meta.get(k) for k in ("transformers_version", "torch_version", "train_rows")} or None,
        )

    @app.post("/v1/predict", response_model=PredictResponse, tags=["model"], dependencies=[Depends(require_key)])
    def predict(req: PredictRequest, p: PredictorLike = Depends(get_predictor)):
        # Sync endpoint -> FastAPI runs it in a worker thread, so the event loop is never blocked by the model.
        if len(req.texts) > settings.max_batch_size:
            raise HTTPException(413, f"batch too large: {len(req.texts)} > {settings.max_batch_size}")
        too_long = [i for i, t in enumerate(req.texts) if len(t) > settings.max_text_chars]
        if too_long:
            raise HTTPException(413, f"texts at positions {too_long[:10]} exceed {settings.max_text_chars} chars")
        with LATENCY.time():
            preds = p.predict(req.texts)
        out = []
        for text, pr in zip(req.texts, preds, strict=True):
            d = pr.to_dict() if hasattr(pr, "to_dict") else dict(pr)
            TEXTS.labels(d["label"], d["language"]).inc()
            out.append(PredictionOut(text=text, **d))
        return PredictResponse(model_version=__version__, predictions=out)

    return app

def app_factory() -> FastAPI:
    return create_app()
