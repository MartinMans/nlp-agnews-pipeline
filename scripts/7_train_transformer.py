import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, set_seed

from src.config import (
    PROCESSED_DIR,
    TRANSFORMER_OUTPUT_DIR,
    RESULTS_DIR,
    RANDOM_STATE,
)
from src.models.transformer import (
    load_model,
    load_tokenizer,
    build_tokenize_fn,
)


def load_splits_as_datasets():
    """
    Load the train and validation splits from CSV into HuggingFace Datasets.
    Expects columns: 'text' and 'label'.
    """
    train_path = PROCESSED_DIR / "ag_news_train.csv"
    val_path = PROCESSED_DIR / "ag_news_val.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


def compute_metrics(eval_pred):
    """
    Compute accuracy and macro F1 for evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def main():
    # Make everything as reproducible as possible
    set_seed(RANDOM_STATE)

    # 1. Load base model + tokenizer
    tokenizer = load_tokenizer()
    model = load_model(num_labels=4)

    # 2. Load splits as Datasets
    train_ds, val_ds = load_splits_as_datasets()

    # 3. Tokenize datasets
    tokenize_fn = build_tokenize_fn(tokenizer, max_length=128)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Ensure 'label' column exists
    if "label" not in train_ds.column_names:
        raise ValueError("Expected 'label' column in dataset.")

    # Keep only the columns we actually need
    keep_cols = ["input_ids", "attention_mask", "label", "text"]
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in keep_cols]
    )
    val_ds = val_ds.remove_columns(
        [c for c in val_ds.column_names if c not in keep_cols]
    )

    # Format for PyTorch
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 4. Training arguments  (requires transformers >= 3.x / 4.x)
    training_args = TrainingArguments(
        output_dir=str(TRANSFORMER_OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        logging_dir=str(TRANSFORMER_OUTPUT_DIR / "logs"),
        report_to="none",  # avoid extra integration noise
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    trainer.train()

    # 7. Final evaluation
    eval_results = trainer.evaluate()
    print("Final evaluation:", eval_results)

    # 8. Save model + tokenizer
    TRANSFORMER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(TRANSFORMER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_OUTPUT_DIR))

    # Save metrics to a text file
    metrics_path = RESULTS_DIR / "transformer_eval_results.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        for k, v in eval_results.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved fine-tuned model and tokenizer to: {TRANSFORMER_OUTPUT_DIR}")
    print(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
