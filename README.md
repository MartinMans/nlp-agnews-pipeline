## AG News NLP Pipeline

This project implements a complete, end-to-end NLP pipeline for classifying news articles from the AG News dataset, a widely used benchmark in text classification.

## Dataset Description

The AG News dataset contains short news articles labeled into four categories:

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
- Clean data handling and preprocessing
- Baseline machine-learning model (TF-IDF + Logistic Regression)
- Improved transformer-based models (e.g., DistilBERT)
- A unified inference interface
- A FastAPI web service for prediction
- Additional containerization and a Streamlit frontend

## Progress So Far

### Step 1 — Load and Inspect Dataset
- Loaded AG News from HuggingFace
- Displayed sample rows, class distribution, and dataset structure
- Confirmed dataset integrity and balance

### Step 2 — Create Train/Validation/Test Splits
- Created splits:
  - Train: 108,000 samples  
  - Validation: 12,000 samples  
  - Test: 7,600 samples  
- Used stratified splitting to preserve class balance  
- Saved all splits as CSV files under `data/processed/`

### Step 3 — Baseline Model (TF-IDF + Logistic Regression)
- Implemented a baseline text classifier using:
  - `TfidfVectorizer` with unigrams + bigrams
  - Multinomial Logistic Regression
- Achieved ~92% validation accuracy
- Saved the trained pipeline under `data/models/`
- Saved evaluation metrics under `data/results/`

### Step 4 — Inference Layer
- Added a reusable inference module at `src/inference/baseline.py`
- Supports:
  - `predict(text)` for single inputs  
  - `predict_batch(list_of_texts)` for batch inference  
- Includes a small test script in `scripts/` that loads the model and runs predictions

## Repository Structure

```
NLP-AGNEWS-PIPELINE/
│
├─ app/ # FastAPI (to be implemented)
│
├─ scripts/ # One-off training/data scripts
│ ├─ load_agnews.py
│ ├─ split_data.py
│ ├─ train_baseline.py
│ └─ test_baseline_inference.py
│
├─ src/ # Reusable project code
│ ├─ config.py
│ ├─ inference/
│ │ └─ baseline.py
│ └─ models/
│   └─ baseline.py
│
├─ data/
│ ├─ processed/
│ │ ├─ ag_news_train.csv
│ │ ├─ ag_news_val.csv
│ │ └─ ag_news_test.csv
│ ├─ models/
│ │ └─ baseline_tfidf_logreg.joblib
│ └─ results/
│   └─ baseline_tfidf_logreg.txt
│
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ venv/
```

## Next Steps

- Implement the FastAPI prediction service
- Build a simple Streamlit frontend that communicates with the API
- Add a transformer-based model and compare performance
