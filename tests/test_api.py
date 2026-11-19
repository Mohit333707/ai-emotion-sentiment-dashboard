"""Real API tests for the AI Emotion & Sentiment Intelligence Dashboard.

Run from the project root:
    pytest -v

Note: the first run downloads the Hugging Face models (~500 MB) and takes a
few minutes. Later runs reuse the cache and are fast. If the models cannot be
downloaded, the API transparently switches to the built-in fallback engine
and these tests still pass (they assert structure, plus sentiment on a very
clear positive sentence, which both engines get right).
"""

import io

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["engine"] in ("transformers", "fallback", "loading")
    assert isinstance(data["models_loaded"], bool)


def test_analyze_positive():
    res = client.post("/analyze", json={"text": "I absolutely love this, it is wonderful!"})
    assert res.status_code == 200
    data = res.json()
    assert data["sentiment"] == "POSITIVE"
    assert 0.0 <= data["sentiment_score"] <= 1.0
    assert data["sentiment_score"] > 0.5
    assert isinstance(data["emotions"], list) and len(data["emotions"]) >= 1
    assert data["top_emotion"] in {e["label"] for e in data["emotions"]}
    assert data["engine"] in ("transformers", "fallback")
    # Emotions are sorted by score, descending.
    scores = [e["score"] for e in data["emotions"]]
    assert scores == sorted(scores, reverse=True)


def test_analyze_negative():
    res = client.post("/analyze", json={"text": "I hate this, it is terrible and awful."})
    assert res.status_code == 200
    assert res.json()["sentiment"] == "NEGATIVE"


def test_analyze_empty_text_rejected():
    res = client.post("/analyze", json={"text": "   "})
    assert res.status_code == 422


def test_analyze_missing_text_rejected():
    res = client.post("/analyze", json={})
    assert res.status_code == 422


def test_analyze_batch_csv():
    csv_content = (
        "text\n"
        "\"I love this product, it is amazing!\"\n"
        "\"This is the worst experience ever, I am furious.\"\n"
        "\"It is an okay product, nothing special.\"\n"
    )
    res = client.post(
        "/analyze-batch",
        files={"file": ("reviews.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["sentiment"] == "POSITIVE"
    assert data["results"][1]["sentiment"] == "NEGATIVE"
    s = data["summary"]
    assert s["positive"] + s["negative"] + s["neutral"] == 3
    assert s["positive"] >= 1 and s["negative"] >= 1


def test_analyze_batch_rejects_non_csv():
    res = client.post(
        "/analyze-batch",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 400
