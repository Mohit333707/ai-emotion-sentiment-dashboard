"""AI engine: sentiment + emotion analysis.

Primary engine: Hugging Face transformers pipelines
  - Sentiment: distilbert-base-uncased-finetuned-sst-2-english
  - Emotion:   j-hartmann/emotion-english-distilroberta-base (7 labels:
               anger, disgust, fear, joy, neutral, sadness, surprise)

Fallback engine: if the Hugging Face models cannot be loaded (no network,
download failure, no torch), a small built-in classifier takes over:
  - Sentiment: TF-IDF + LogisticRegression trained on a labelled dataset
    bundled in this file (a real trained model, not hard-coded rules).
  - Emotion: a transparent keyword-lexicon scorer.
The response always reports which engine produced it, so results are never
misrepresented as coming from the transformer models.
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ---------------------------------------------------------------------------
# Fallback: small labelled dataset for the sentiment classifier.
# ---------------------------------------------------------------------------
_FALLBACK_POSITIVE = [
    "I love this", "this is amazing", "absolutely wonderful", "great job",
    "I am so happy", "fantastic experience", "best day ever", "I really enjoyed it",
    "excellent work", "so proud of you", "this made my day", "brilliant idea",
    "I feel great", "wonderful news", "I am thrilled", "perfect, thank you",
    "love it", "so good", "incredible", "delightful", "I am grateful",
    "such a joy", "very pleased", "impressive", "I adore this", "superb",
    "happy birthday", "congratulations, well done", "this is the best",
    "I feel wonderful", "amazing product", "highly recommend", "five stars",
    "so excited", "beautiful", "I am delighted", "great news", "awesome",
    "I love you", "thank you so much", "this is lovely", "outstanding",
    "I am so pleased", "wonderful experience", "really good", "I feel lucky",
    "such a nice surprise", "brilliant work", "I am overjoyed", "fantastic",
    "this is great", "I am smiling", "very happy with this", "top class",
    "I feel blessed", "amazing job", "so much fun", "I love it so much",
    "this is excellent", "wonderful day", "I am so glad", "perfect",
]
_FALLBACK_NEGATIVE = [
    "I hate this", "this is terrible", "awful experience", "so disappointed",
    "I am angry", "worst day ever", "this is horrible", "I feel sad",
    "very bad", "I am frustrated", "disgusting", "I am upset",
    "this is a disaster", "I am so unhappy", "poor quality", "waste of time",
    "I regret this", "terrible service", "I am annoyed", "this is bad",
    "I feel awful", "so depressing", "I am heartbroken", "this sucks",
    "one star", "never again", "I am furious", "this is useless",
    "I am stressed", "so boring", "I hate it", "this is painful",
    "very disappointed", "I feel terrible", "awful job", "I am miserable",
    "this is the worst", "I am so tired of this", "horrible", "I am scared",
    "this is unfair", "I feel sick", "bad news", "I am worried",
    "this is broken", "I am so sad", "pathetic", "I am disgusted",
    "this is a nightmare", "I am crying", "so angry", "I feel hopeless",
    "this is dreadful", "I am annoyed", "terrible", "I feel lonely",
    "this is disappointing", "I am in pain", "awful news", "I am fed up",
]

# Fallback emotion lexicon: word -> emotion label.
_FALLBACK_EMOTION_LEXICON = {
    "joy": ["happy", "joy", "love", "great", "wonderful", "amazing", "fantastic",
            "delight", "excited", "smile", "laugh", "fun", "beautiful", "glad",
            "pleased", "thrilled", "awesome", "brilliant", "celebrat", "birthday"],
    "anger": ["angry", "furious", "hate", "annoyed", "frustrat", "mad", "rage",
              "outrage", "irritat", "disgust", "pathetic", "ridiculous", "unfair"],
    "fear": ["afraid", "scared", "fear", "worried", "anxious", "nervous", "panic",
             "terrified", "dread", "horror", "nightmare", "worry"],
    "sadness": ["sad", "unhappy", "depress", "cry", "heartbreak", "lonely",
                "miserable", "grief", "sorrow", "tears", "hopeless", "pain"],
    "surprise": ["wow", "surprise", "shocked", "amazing", "unbelievable", "incredible",
                 "astonish", "unexpected", "suddenly", "oh my"],
    "love": ["love", "adore", "affection", "darling", "sweetheart", "beloved",
             "cherish", "romantic", "dear"],
    "neutral": [],
}
_EMOTION_LABELS = ["joy", "anger", "fear", "sadness", "surprise", "love", "neutral"]


class Analyzer:
    """Sentiment + emotion analyzer with a transparent fallback engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self.engine = "loading"
        self._sentiment_pipe = None
        self._emotion_pipe = None
        # Fallback pieces (built lazily only if needed).
        self._fallback_clf = None
        self._fallback_vec = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> str:
        """Load the Hugging Face models; fall back cleanly on any failure.

        Returns the engine name in use: 'transformers' or 'fallback'.
        Safe to call repeatedly; loading happens at most once.
        """
        with self._lock:
            if self._loaded:
                return self.engine
            try:
                from transformers import pipeline

                logger.info("Loading sentiment model: %s", SENTIMENT_MODEL)
                self._sentiment_pipe = pipeline(
                    "sentiment-analysis", model=SENTIMENT_MODEL
                )
                logger.info("Loading emotion model: %s", EMOTION_MODEL)
                self._emotion_pipe = pipeline(
                    "text-classification",
                    model=EMOTION_MODEL,
                    top_k=None,
                )
                self.engine = "transformers"
                logger.info("Transformers engine ready.")
            except Exception as exc:  # noqa: BLE001 - any load failure -> fallback
                logger.warning("Could not load Hugging Face models (%s). "
                               "Switching to fallback engine.", exc)
                self._build_fallback()
                self.engine = "fallback"
            self._loaded = True
            return self.engine

    def _build_fallback(self) -> None:
        """Train the small built-in sentiment classifier (TF-IDF + LogReg)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = _FALLBACK_POSITIVE + _FALLBACK_NEGATIVE
        labels = [1] * len(_FALLBACK_POSITIVE) + [0] * len(_FALLBACK_NEGATIVE)
        self._fallback_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self._fallback_vec.fit_transform(texts)
        self._fallback_clf = LogisticRegression(max_iter=1000)
        self._fallback_clf.fit(X, labels)
        logger.info("Fallback sentiment classifier trained on %d examples.", len(texts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, text: str) -> dict:
        """Analyse one text. Returns a dict matching AnalyzeResponse."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Text must not be empty.")
        self.load()
        if self.engine == "transformers":
            return self._analyze_transformers(text)
        return self._analyze_fallback(text)

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyse many texts, preserving order."""
        self.load()
        if self.engine == "transformers":
            return self._analyze_batch_transformers(texts)
        return [self._analyze_fallback(t) for t in texts]

    # ------------------------------------------------------------------
    # Transformers engine
    # ------------------------------------------------------------------
    def _all_scores(self, pipe, text: str) -> list[dict]:
        """Call a text-classification pipeline asking for all label scores."""
        try:
            out = pipe(text, top_k=None)
        except TypeError:
            out = pipe(text, return_all_scores=True)
        # transformers v5 returns a list of dicts; older versions a list of lists.
        if out and isinstance(out[0], list):
            out = out[0]
        return out

    def _analyze_transformers(self, text: str) -> dict:
        sent = self._sentiment_pipe(text)[0]
        emo_raw = self._all_scores(self._emotion_pipe, text)
        emotions = sorted(
            ({"label": e["label"].lower(), "score": round(float(e["score"]), 4)}
             for e in emo_raw),
            key=lambda d: d["score"],
            reverse=True,
        )
        return {
            "text": text,
            "sentiment": sent["label"].upper(),
            "sentiment_score": round(float(sent["score"]), 4),
            "emotions": emotions,
            "top_emotion": emotions[0]["label"] if emotions else "neutral",
            "engine": "transformers",
        }

    def _analyze_batch_transformers(self, texts: list[str]) -> list[dict]:
        sents = self._sentiment_pipe(texts)
        emos = self._emotion_pipe(texts, top_k=None)
        results = []
        for text, sent, emo_raw in zip(texts, sents, emos):
            if emo_raw and isinstance(emo_raw[0], list):
                emo_raw = emo_raw[0]
            emotions = sorted(
                ({"label": e["label"].lower(), "score": round(float(e["score"]), 4)}
                 for e in emo_raw),
                key=lambda d: d["score"],
                reverse=True,
            )
            results.append({
                "text": text,
                "sentiment": sent["label"].upper(),
                "sentiment_score": round(float(sent["score"]), 4),
                "emotions": emotions,
                "top_emotion": emotions[0]["label"] if emotions else "neutral",
                "engine": "transformers",
            })
        return results

    # ------------------------------------------------------------------
    # Fallback engine
    # ------------------------------------------------------------------
    def _analyze_fallback(self, text: str) -> dict:
        proba = self._fallback_clf.predict_proba(
            self._fallback_vec.transform([text])
        )[0]
        # classes are [0 (negative), 1 (positive)]
        pos = float(proba[1])
        sentiment = "POSITIVE" if pos >= 0.5 else "NEGATIVE"
        score = pos if sentiment == "POSITIVE" else 1.0 - pos

        lowered = text.lower()
        counts = {label: 0 for label in _EMOTION_LABELS}
        for label, words in _FALLBACK_EMOTION_LEXICON.items():
            for w in words:
                if w in lowered:
                    counts[label] += 1
        total = sum(counts.values())
        if total == 0:
            counts["neutral"] = 1
            total = 1
        emotions = sorted(
            ({"label": label, "score": round(count / total, 4)}
             for label, count in counts.items()),
            key=lambda d: d["score"],
            reverse=True,
        )
        return {
            "text": text,
            "sentiment": sentiment,
            "sentiment_score": round(score, 4),
            "emotions": emotions,
            "top_emotion": emotions[0]["label"],
            "engine": "fallback",
        }


# Single shared instance used by the API.
analyzer = Analyzer()
