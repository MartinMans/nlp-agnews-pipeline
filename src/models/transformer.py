from typing import Dict, Any

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import ID2LABEL, LABEL2ID, TRANSFORMER_MODEL_NAME


def load_tokenizer():
    """
    Load the tokenizer for the base transformer model.
    """
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    return tokenizer


def load_model(num_labels: int = 4):
    """
    Load a DistilBERT-based sequence classification model
    configured for the AG News label space.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def build_tokenize_fn(tokenizer, max_length: int = 128):
    """
    Return a function suitable for Dataset.map that tokenizes
    a batch of examples with a 'text' field.
    """

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return tokenize_batch
