from dataclasses import dataclass
import pytest
from fastapi.testclient import TestClient
from xlmr_sentiment.api.app import Settings, create_app
from xlmr_sentiment.text import profile_script

@dataclass
class _Pred:
    label: str
    confidence: float
    scores: dict
    language: str
    code_mixed: bool

    def to_dict(self):
        return self.__dict__.copy()

class FakePredictor:
    labels = ["negative", "neutral", "positive"]
    device = "cpu"
    quantized = False
    meta = {"base_model": "fake", "validation_metrics": {"eval_macro_f1": 0.5}}

    def predict(self, texts):
        out = []
        for t in texts:
            label = "negative" if ("bad" in t or "برا" in t) else "positive"
            scores = {lbl: (0.9 if lbl == label else 0.05) for lbl in self.labels}
            prof = profile_script(t)
            out.append(_Pred(label, 0.9, scores, prof.language, prof.code_mixed))
        return out

@pytest.fixture
def client():
    with TestClient(create_app(FakePredictor(), Settings(max_batch_size=3, max_text_chars=50))) as c:
        yield c

def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    assert "sentiment_requests_total" in client.get("/metrics").text

def test_predict_batch(client):
    r = client.post("/v1/predict", json={"texts": ["good movie", "یہ برا ہے", "bad hai yaar"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["label"] for p in preds] == ["positive", "negative", "negative"]
    assert preds[1]["language"] == "ur"
    assert "X-Request-ID" in r.headers

@pytest.mark.parametrize("payload,code", [
    ({"texts": []}, 422),
    ({"texts": ["  "]}, 422),
    ({"texts": ["a", "b", "c", "d"]}, 413),
    ({"texts": ["x" * 51]}, 413),
    ({}, 422),
])
def test_predict_validation(client, payload, code):
    assert client.post("/v1/predict", json=payload).status_code == code

def test_model_info(client):
    j = client.get("/v1/model").json()
    assert j["labels"] == FakePredictor.labels and j["validation_macro_f1"] == 0.5

def test_api_key_enforced():
    app = create_app(FakePredictor(), Settings(api_key="s3cret"))
    with TestClient(app) as c:
        assert c.post("/v1/predict", json={"texts": ["hi"]}).status_code == 401
        assert c.post("/v1/predict", json={"texts": ["hi"]}, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/v1/predict", json={"texts": ["hi"]}, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200  # ops endpoints stay open

def test_not_ready_when_model_missing(tmp_path):
    app = create_app(None, Settings(model_dir=str(tmp_path / "missing")))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/v1/predict", json={"texts": ["hi"]}).status_code == 503
