"""
Utility script for inspecting the AG News dataset structure, columns,
sample rows, and label distribution.
"""

from datasets import load_dataset
import pandas as pd


def main():
    """Load the AG News dataset and print basic statistics and samples."""
    ds = load_dataset("ag_news")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain columns:", train_df.columns.tolist())
    print("\nSample rows:")
    print(train_df.head(5))

    print("\nLabel distribution:")
    print(train_df["label"].value_counts())


if __name__ == "__main__":
    main()
