"""AI Emotion & Sentiment Intelligence Dashboard — FastAPI backend.

Endpoints:
  GET  /health          - health check + which engine is active
  POST /analyze        - analyse a single text
  POST /analyze-batch  - upload a CSV (needs a `text` column) and analyse every row
  GET  /                - the dashboard frontend (static files)

Run from the project root:
  uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import csv
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .analyzer import analyzer
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchResponse,
    BatchSummary,
    HealthResponse,
)

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 5000
MAX_BATCH_ROWS = 500


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the AI models once at startup so the first request is fast."""
    engine = analyzer.load()
    logger.info("Backend ready, engine=%s", engine)
    yield


app = FastAPI(
    title="AI Emotion & Sentiment Intelligence Dashboard",
    description="Real-time sentiment analysis and emotion detection powered by "
                "Hugging Face transformer models.",
    version="1.0.0",
    lifespan=lifespan,
)

# The dashboard is served from the same origin, but CORS is kept open so the
# frontend can also be opened as a plain file or from another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Health check: is the API up, and which engine is active?"""
    engine = analyzer.engine if analyzer._loaded else "loading"
    return HealthResponse(
        status="ok",
        engine=engine,
        models_loaded=analyzer._loaded,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse a single text for sentiment and emotion."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Text is too long (max {MAX_TEXT_LENGTH} characters).",
        )
    try:
        return AnalyzeResponse(**analyzer.analyze(text))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/analyze-batch", response_model=BatchResponse, tags=["analysis"])
async def analyze_batch(file: UploadFile = File(...)) -> BatchResponse:
    """Analyse a CSV file. It must have a `text` column (otherwise the first
    column is used). At most 500 rows are processed per upload."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="CSV must be UTF-8 encoded."
        ) from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    # Prefer a `text` column (case-insensitive); otherwise use the first column.
    fieldnames = [f for f in reader.fieldnames if f]
    text_col = next((f for f in fieldnames if f.strip().lower() == "text"),
                    fieldnames[0])

    texts = []
    for row in reader:
        value = (row.get(text_col) or "").strip()
        if value:
            texts.append(value)
        if len(texts) >= MAX_BATCH_ROWS:
            break

    if not texts:
        raise HTTPException(
            status_code=400,
            detail=f"No usable text found in column '{text_col}'.",
        )

    try:
        results = [AnalyzeResponse(**r) for r in analyzer.analyze_batch(texts)]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch analysis failed.")
        raise HTTPException(status_code=500, detail=f"Batch failed: {exc}") from exc

    summary = BatchSummary()
    for r in results:
        if r.sentiment == "POSITIVE":
            summary.positive += 1
        elif r.sentiment == "NEGATIVE":
            summary.negative += 1
        else:
            summary.neutral += 1
        summary.top_emotions[r.top_emotion] = (
            summary.top_emotions.get(r.top_emotion, 0) + 1
        )

    return BatchResponse(
        total=len(results),
        results=results,
        summary=summary,
        engine=results[0].engine if results else analyzer.engine,
    )


# --- Frontend: serve the dashboard from the same server -------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
