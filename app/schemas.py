# app/schemas.py
from typing import List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    texts: List[str]


class Prediction(BaseModel):
    text: str
    label_id: int
    label: str
    probs: List[float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
