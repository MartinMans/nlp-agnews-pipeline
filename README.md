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
- A Streamlit frontend for interactive use  
- Optional containerization for deployment  

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
  - `TfidfVectorizer` (unigrams + bigrams)  
  - Multinomial Logistic Regression  
- Achieved ~92% validation accuracy  
- Saved the trained model under `data/models/`  
- Saved evaluation metrics under `data/results/`  

### Step 4 — Inference Layer (Baseline)
- Added a reusable inference module at `src/inference/baseline.py`  
- Provides:  
  - `predict(text)` for single inputs  
  - `predict_batch(list_of_texts)` for batch inference  
- Includes a test script in `scripts/` to verify predictions  

### Step 5 — FastAPI Backend
- Implemented a FastAPI application with two endpoints:  
  - `/health` — service status  
  - `/predict` — baseline batch predictions  
- Integrated backend with the baseline inference module  
- API documentation available through Swagger (`/docs`)  

### Step 6 — Streamlit Frontend
- Built a fully functional Streamlit UI for interacting with the model  
- Supports single and multi-line classification (one text per line)  
- Displays predicted labels and probability distributions using Altair  
- Communicates with the FastAPI backend via HTTP requests  

### Step 7 — Transformer Fine-Tuning (DistilBERT)
- Fine-tuned `distilbert-base-uncased` on AG News  
- Achieved ~94.6% validation accuracy and macro F1  
- Saved the model, tokenizer, and checkpoints under  
  `data/models/transformer_distilbert/`  
- Stored evaluation results in `eval_results.txt`

### Step 8 — Transformer Inference
- Added inference module at `src/inference/transformer.py`  
- Provides:  
  - `predict(text)`  
  - `predict_batch(list_of_texts)`  
- Loads fine-tuned DistilBERT with `AutoModelForSequenceClassification`  
- Mirrors the baseline inference interface  

### Step 9 — FastAPI Transformer Endpoint
- Added a second prediction endpoint:  
  - `/predict-transformer`  
- Returns identical JSON structure as baseline inference  
- Allows the frontend to switch models seamlessly  

### Step 10 — Streamlit Model Selector
- Updated the Streamlit UI to support model selection:  
  - **Baseline (TF-IDF + Logistic Regression)**  
  - **Transformer (DistilBERT)**  
- Same frontend, two different backends  
- Makes it easy to compare performance and behavior  

## Repository Structure

```
NLP-AGNEWS-PIPELINE/
│
├─ app/                      # FastAPI application
│  ├─ __init__.py
│  ├─ main.py
│  └─ schemas.py
│
├─ frontend/                 # Streamlit user interface
│  └─ streamlit_app.py
│
├─ scripts/                  # One-off training/data scripts
│  ├─ load_agnews.py
│  ├─ split_and_save.py
│  ├─ train_baseline.py
│  ├─ test_baseline_inference.py
│  └─ train_transformer.py
│
├─ src/                      # Reusable project code
│  ├─ __init__.py
│  ├─ config.py
│  ├─ inference/
│  │  ├─ __init__.py
│  │  ├─ baseline.py
│  │  └─ transformer.py
│  └─ models/
│     ├─ __init__.py
│     ├─ baseline.py
│     └─ transformer.py
│
├─ data/                     # (ignored by Git) local data and models
│  ├─ processed/             # train/val/test CSV splits
│  ├─ models/
│  │  ├─ baseline_tfidf_logreg.joblib
│  │  └─ transformer_distilbert/
│  │     ├─ config.json
│  │     ├─ model.safetensors
│  │     ├─ tokenizer.json
│  │     └─ ...              # checkpoints, training args, etc.
│  └─ results/
│     ├─ baseline_tfidf_logreg.txt
│     └─ transformer_eval_results.txt
│
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ venv/                     # (ignored) local virtual environment
```

## Next Steps

- Containerize the FastAPI and Streamlit applications using Docker  
- Provide a lightweight deployment setup (Docker Compose or a single-run script)  
- Maybe use a cloud provider? Not sure yet.

