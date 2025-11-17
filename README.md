## AG News NLP Pipeline

This project implements a complete, end-to-end NLP pipeline for classifying news articles from the AG News dataset, a widely used benchmark in text classification.

## Dataset Description

The AG News dataset contains short news articles labeled into four classes:

| Label ID | Category |
|----------|----------|
| 0        | World    |
| 1        | Sports   |
| 2        | Business |
| 3        | Sci/Tech |

Each example consists of a short text (title + description) and a corresponding label indicating its topic.

Dataset sources:
- HuggingFace: https://huggingface.co/datasets/sh0416/ag_news
- Kaggle (variant): https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

## Project Goal

The aim of this project is to build a fully deployable NLP classification system, including:
- Clean data handling (loading, splitting, preprocessing)
- Baseline machine-learning model (TF-IDF + Logistic Regression)
- Improved transformer-based models (e.g., DistilBERT)
- A unified inference interface
- A FastAPI web service for prediction
- Additional containerization and frontend UI

## Progress So Far

### Step 1 — Load and Inspect Dataset
- Created a script to load AG News from HuggingFace
- Printed sample rows, column names, and label distribution
- Confirmed dataset integrity and class balance

### Step 2 — Create Train/Validation/Test Splits
- Splits:
  - Train: 108,000 samples
  - Validation: 12,000 samples
  - Test: 7,600 samples
- Used train_test_split with stratification to preserve label balance
- Saved splits as CSV files under data/processed/

## Next Steps

- Implement baseline model: TF-IDF + Logistic Regression
- Train, evaluate, and save the model as a Scikit-Learn pipeline
- Begin setting up the inference layer and API

## Repository Structure (current)

```
NLP-AGNEWS-PIPELINE/
│
├─ data/
│  └─ processed/
│     ├─ ag_news_train.csv
│     ├─ ag_news_val.csv
│     └─ ag_news_test.csv
│
├─ venv/
├─ README.md
├─ requirements.txt
├─ step1_load_agnews.py
└─ step2_split_and_save.py
```
