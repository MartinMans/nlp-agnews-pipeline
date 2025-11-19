from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np

from src.config import MODELS_DIR, ID2LABEL


MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
_model = None  # cached model instance


def load_model():
    """Load the trained baseline model (cached after first load)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(text: str) -> Dict[str, Any]:
    """
    Predict the label for a single news text.

    Returns a dict with:
    - label_id: int
    - label: str (human-readable)
    - probs: list[float] of length 4 (probabilities per class)
    """
    model = load_model()

    probs = model.predict_proba([text])[0]  # shape: (4,)
    probs = np.asarray(probs)
    label_id = int(probs.argmax())
    label = ID2LABEL[label_id]

    return {
        "label_id": label_id,
        "label": label,
        "probs": probs.tolist(),
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict labels for a list of news texts.

    Returns a list of dicts, each containing:
    - text
    - label_id
    - label
    - probs
    """
    if not texts:
        return []

    model = load_model()

    probs_matrix = model.predict_proba(texts)  # shape: (N, 4)
    results: List[Dict[str, Any]] = []

    for text, probs in zip(texts, probs_matrix):
        probs = np.asarray(probs)
        label_id = int(probs.argmax())
        label = ID2LABEL[label_id]

        results.append(
            {
                "text": text,
                "label_id": label_id,
                "label": label,
                "probs": probs.tolist(),
            }
        )

    return results
