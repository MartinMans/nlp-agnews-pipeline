from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import TRANSFORMER_OUTPUT_DIR

_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_and_tokenizer():
    """Lazy-load the fine-tuned transformer model + tokenizer (cached)."""
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        if not TRANSFORMER_OUTPUT_DIR.exists():
            raise FileNotFoundError(
                f"Transformer model directory not found at {TRANSFORMER_OUTPUT_DIR}"
            )

        _tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_OUTPUT_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_OUTPUT_DIR
        )
        _model.to(_device)
        _model.eval()

    return _model, _tokenizer


def predict(text: str) -> Dict[str, Any]:
    """
    Predict the label for a single news text using the fine-tuned transformer.
    """
    model, tokenizer = _load_model_and_tokenizer()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    encoded = {k: v.to(_device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  # shape: (1, num_labels)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(np.argmax(probs))
    # model.config.id2label was set during training
    label = model.config.id2label.get(label_id, str(label_id))

    return {
        "label_id": label_id,
        "label": label,
        "probs": probs.tolist(),
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict labels for a batch of texts using the fine-tuned transformer.
    """
    if not texts:
        return []

    model, tokenizer = _load_model_and_tokenizer()

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    encoded = {k: v.to(_device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  # shape: (N, num_labels)
        probs_matrix = torch.softmax(logits, dim=-1).cpu().numpy()

    results: List[Dict[str, Any]] = []
    id2label = model.config.id2label

    for text, probs in zip(texts, probs_matrix):
        label_id = int(np.argmax(probs))
        label = id2label.get(label_id, str(label_id))

        results.append(
            {
                "text": text,
                "label_id": label_id,
                "label": label,
                "probs": probs.tolist(),
            }
        )

    return results
