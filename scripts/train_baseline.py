"""
Train and evaluate the baseline TF-IDF + Logistic Regression model
on the AG News dataset, then save the model and validation results.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
from src.models.baseline import build_baseline_pipeline


def load_splits():
    """Load preprocessed train and validation splits from disk."""
    train_path = PROCESSED_DIR / "ag_news_train.csv"
    val_path = PROCESSED_DIR / "ag_news_val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    return train_df, val_df


def main():
    """Train the baseline model, evaluate it, and save outputs."""
    # Load training and validation data
    train_df, val_df = load_splits()

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["label"].tolist()

    X_val = val_df["text"].astype(str).tolist()
    y_val = val_df["label"].tolist()

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Build and train the baseline model
    model = build_baseline_pipeline()
    print("Training baseline TF-IDF + Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=4)

    metrics_text = []
    metrics_text.append("Baseline TF-IDF + Logistic Regression\n")
    metrics_text.append(f"Validation accuracy: {acc:.4f}\n")
    metrics_text.append("\nClassification report:\n")
    metrics_text.append(report)
    metrics_text = "".join(metrics_text)

    print("\n" + "=" * 60)
    print(metrics_text)
    print("=" * 60 + "\n")

    # Save trained model and evaluation metrics
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "baseline_tfidf_logreg.joblib"
    results_path = RESULTS_DIR / "baseline_tfidf_logreg.txt"

    joblib.dump(model, model_path)
    results_path.write_text(metrics_text, encoding="utf-8")

    print(f"Saved model to:   {model_path}")
    print(f"Saved metrics to: {results_path}")


if __name__ == "__main__":
    main()
