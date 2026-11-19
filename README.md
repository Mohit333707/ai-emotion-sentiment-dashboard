# 🧠 AI Emotion & Sentiment Intelligence Dashboard

A full-stack AI web app that reads any text — a review, a tweet, a message — and tells you **what it feels**: overall sentiment (positive / negative) with confidence, plus a full **emotion breakdown** (joy, anger, fear, sadness, surprise, love, neutral) shown as bars. It also analyses a whole **CSV file** of texts in one go and summarises the results.

Built with **FastAPI + Hugging Face transformers**, with a clean single-page dashboard (plain HTML/CSS/JS — no build step).

---

## ✨ Features

- **Single-text analysis** — type or paste any text, get sentiment + confidence + emotion bars.
- **Batch CSV analysis** — upload a CSV with a `text` column (up to 500 rows) and get per-row results plus a summary (counts + top emotion).
- **History** — your recent analyses are saved in the browser (localStorage).
- **Live backend status** — the dashboard shows whether the API is online and which engine is running.
- **Sample CSV** — one click downloads a 10-row sample file so you can try batch mode instantly.

## 🧠 How the AI works

| Task | Primary model (Hugging Face) |
|---|---|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| Emotion (7 labels) | `j-hartmann/emotion-english-distilroberta-base` |

Both models run **locally on CPU** — no API key, no data leaves your machine.

**Honest fallback:** if the Hugging Face models cannot be downloaded (offline machine, blocked network), the backend automatically switches to a built-in fallback engine — a TF-IDF + Logistic Regression classifier trained on a labelled dataset bundled in `backend/app/analyzer.py`, plus a transparent keyword-lexicon emotion scorer. Every API response includes an `engine` field (`"transformers"` or `"fallback"`), so results are never misrepresented.

## 🛠️ Tech stack

- **Backend:** Python 3.10+, FastAPI, Pydantic, Uvicorn, Hugging Face `transformers` + PyTorch (CPU), scikit-learn (fallback)
- **Frontend:** plain HTML, CSS, JavaScript — no framework, no build step
- **Tests:** pytest + FastAPI TestClient

## 📁 Project structure

```
ai-emotion-sentiment-dashboard/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py        # FastAPI app: routes, CORS, CSV upload, static frontend
│       ├── analyzer.py    # Model loading, inference, honest fallback engine
│       └── models.py      # Pydantic request/response schemas
├── frontend/
│   └── index.html         # The whole dashboard (HTML + CSS + JS)
├── tests/
│   └── test_api.py        # Real API tests (health, analyse, batch, errors)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd ai-emotion-sentiment-dashboard

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (CPU torch — much smaller download)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> First run downloads the two models from Hugging Face (~500 MB) and takes a few minutes. Later runs reuse the cache and start fast.

## ▶️ Run it

```bash
# From the project root, with the venv active:
uvicorn backend.app.main:app --reload --port 8000
```

Then open **http://localhost:8000** — the dashboard and the API run on the same server.

Interactive API docs (Swagger UI): **http://localhost:8000/docs**

## 🔌 API

### `GET /health`
```json
{ "status": "ok", "engine": "transformers", "models_loaded": true }
```

### `POST /analyze`
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'
```
```json
{
  "text": "I absolutely love this product!",
  "sentiment": "POSITIVE",
  "sentiment_score": 0.9998,
  "emotions": [
    { "label": "joy", "score": 0.9684 },
    { "label": "love", "score": 0.0241 },
    { "label": "surprise", "score": 0.0031 }
  ],
  "top_emotion": "joy",
  "engine": "transformers"
}
```

### `POST /analyze-batch`
Upload a CSV (`multipart/form-data`, field `file`). The CSV needs a `text` column (otherwise the first column is used); max 500 rows.

```bash
curl -X POST http://localhost:8000/analyze-batch \
  -F "file=@reviews.csv"
```
```json
{
  "total": 3,
  "results": [ { "...": "one AnalyzeResponse per row" } ],
  "summary": { "positive": 2, "negative": 1, "neutral": 0, "top_emotions": { "joy": 2, "anger": 1 } },
  "engine": "transformers"
}
```

## 🧪 Tests

```bash
pytest -v
```

The suite hits the real endpoints (health, single analysis, batch CSV, error cases). The first run loads the models, so give it a few minutes.

## 📝 Notes

- The emotion model returns 7 labels: `anger, disgust, fear, joy, neutral, sadness, surprise`. The fallback engine adds `love` from its lexicon.
- Batch mode processes at most 500 rows per upload; empty rows are skipped.
- No data is sent to any external service at inference time — everything runs locally.

## 📄 License

MIT — see [LICENSE](LICENSE).
