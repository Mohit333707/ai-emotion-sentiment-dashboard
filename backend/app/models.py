"""Pydantic schemas for the AI Emotion & Sentiment Intelligence Dashboard."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Single-text analysis request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to analyse for sentiment and emotion.",
        examples=["I absolutely love this product, it made my day!"],
    )


class EmotionScore(BaseModel):
    """One emotion label with its confidence score (0-1)."""

    label: str = Field(..., description="Emotion label, e.g. joy, anger, sadness.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0 to 1.")


class AnalyzeResponse(BaseModel):
    """Result of a single-text analysis."""

    text: str = Field(..., description="The analysed text (echoed back).")
    sentiment: str = Field(..., description="Overall sentiment: POSITIVE, NEGATIVE or NEUTRAL.")
    sentiment_score: float = Field(..., ge=0.0, le=1.0, description="Confidence of the sentiment label.")
    emotions: list[EmotionScore] = Field(
        ..., description="Emotion breakdown, sorted by score descending."
    )
    top_emotion: str = Field(..., description="The highest-scoring emotion label.")
    engine: str = Field(
        ...,
        description="Which engine produced the result: 'transformers' (Hugging Face models) "
        "or 'fallback' (built-in trained classifier).",
    )


class BatchSummary(BaseModel):
    """Aggregate counts for a batch run."""

    positive: int = 0
    negative: int = 0
    neutral: int = 0
    top_emotions: dict[str, int] = Field(
        default_factory=dict, description="How many rows had each emotion as their top emotion."
    )


class BatchResponse(BaseModel):
    """Result of a CSV batch analysis."""

    total: int = Field(..., description="Number of rows analysed.")
    results: list[AnalyzeResponse] = Field(..., description="Per-row results, in CSV order.")
    summary: BatchSummary = Field(..., description="Aggregate counts across all rows.")
    engine: str = Field(..., description="Engine used for the batch (see AnalyzeResponse).")


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(..., description="'ok' when the API is up.")
    engine: str = Field(..., description="Current engine: 'transformers', 'fallback' or 'loading'.")
    models_loaded: bool = Field(..., description="True once the AI models (or fallback) are ready.")
