"""
Build the TF-IDF + Logistic Regression pipeline used as the baseline
classifier for the AG News dataset.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def build_baseline_pipeline() -> Pipeline:
    """
    Construct the TF-IDF + Logistic Regression pipeline for baseline classification.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline
