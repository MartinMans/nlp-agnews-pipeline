# app/main.py
from typing import List

from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse, Prediction
from src.inference.baseline import predict_batch


app = FastAPI(
    title="AG News Topic Classifier",
    version="0.1.0",
    description="FastAPI service for classifying AG News articles using a baseline TF-IDF + Logistic Regression model.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # Call existing batch predictor
    results = predict_batch(request.texts)

    predictions: List[Prediction] = [
        Prediction(
            text=r["text"],
            label_id=r["label_id"],
            label=r["label"],
            probs=r["probs"],
        )
        for r in results
    ]

    return PredictResponse(predictions=predictions)
