"""
Script to load the AG News dataset, create a train/validation split,
and save the processed CSV files into data/processed/.
"""

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_STATE = 1


def main():
    """Prepare AG News splits and write them to processed CSV files."""
    # Load the dataset
    ds = load_dataset("ag_news")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # Create validation split from the training data
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")

    # Ensure processed data directory exists
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed CSVs
    train_split.to_csv(processed_dir / "ag_news_train.csv", index=False)
    val_split.to_csv(processed_dir / "ag_news_val.csv", index=False)
    test_df.to_csv(processed_dir / "ag_news_test.csv", index=False)

    print("Saved:")
    print(f"- {processed_dir / 'ag_news_train.csv'}")
    print(f"- {processed_dir / 'ag_news_val.csv'}")
    print(f"- {processed_dir / 'ag_news_test.csv'}")


if __name__ == "__main__":
    main()
