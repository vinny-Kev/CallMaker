import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from feature_engineer import generate_all_features

load_dotenv()

app = FastAPI()

# Load models and weights (stub)
models = {}
ensemble_weights = {}
try:
    models['rf'] = joblib.load('data/models/rf.pkl')
    models['catboost'] = joblib.load('data/models/catboost.pkl')
    models['lr'] = joblib.load('data/models/lr.pkl')
    ensemble_weights = joblib.load('data/models/ensemble_weights.pkl')
except Exception:
    pass

class PredictRequest(BaseModel):
    candles: list  # List of dicts with OHLCV

@app.post('/predict')
def predict(req: PredictRequest):
    df = pd.DataFrame(req.candles)
    df = generate_all_features(df)
    X = df.iloc[[-1]]  # Use last row
    preds = [models[m].predict_proba(X)[0][1] for m in ['rf','catboost','lr'] if m in models]
    weights = [ensemble_weights.get(m, 1/3) for m in ['rf','catboost','lr']]
    if preds:
        proba = float(sum(p*w for p,w in zip(preds,weights)) / sum(weights))
    else:
        proba = 0.5
    signal = 'Buy' if proba > 0.5 else 'Sell'
    # TODO: Load and return model stats
    return {'signal': signal, 'confidence': proba, 'precision': None, 'accuracy': None}

@app.post('/retrain')
def retrain():
    # TODO: Trigger trainer.py
    return {'status': 'Retrain triggered'}
