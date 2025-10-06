"""
Backend API - Proxy to ML Prediction Service
Connects the Streamlit frontend to the ML prediction API
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from typing import Optional

load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(
    title="Crypto Trading Backend",
    description="Backend service for crypto trading dashboard"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction API URL
PREDICTION_API_URL = os.getenv('PREDICTION_API_URL', 'http://localhost:8000')


class PredictRequest(BaseModel):
    """Prediction request from frontend"""
    symbol: str = "BTCUSDT"
    interval: str = "30s"


class PredictResponse(BaseModel):
    """Prediction response to frontend"""
    signal: str
    confidence: float
    prediction_label: str
    probabilities: dict
    current_price: float
    expected_movement: Optional[float] = None
    next_periods: list = []


@app.get('/')
def health_check():
    """Health check"""
    return {
        'status': 'healthy',
        'service': 'backend',
        'prediction_api': PREDICTION_API_URL
    }


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Get prediction from ML API
    Converts response format for frontend compatibility
    """
    try:
        # Call prediction API
        response = requests.post(
            f"{PREDICTION_API_URL}/predict",
            json={
                'symbol': req.symbol,
                'interval': req.interval,
                'use_live_data': True
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Prediction API error: {response.text}"
            )
        
        data = response.json()
        
        # Convert prediction to signal
        prediction = data['prediction']
        if prediction == 1:
            signal = 'BUY'
        elif prediction == 2:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return PredictResponse(
            signal=signal,
            confidence=data['confidence'],
            prediction_label=data['prediction_label'],
            probabilities=data['probabilities'],
            current_price=data['current_price'],
            expected_movement=data.get('expected_movement'),
            next_periods=data.get('next_periods', [])
        )
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Prediction API not available. Please start the prediction service."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Prediction API timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post('/retrain')
def retrain():
    """
    Trigger model retraining
    """
    try:
        # Import and run training pipeline
        from train_pipeline import MLPipeline
        
        # Run training in background (in production, use Celery or similar)
        pipeline = MLPipeline(symbol="BTCUSDT", interval="30s")
        
        # Start training (this will take time)
        # In production, this should be async/background task
        results = pipeline.run_full_pipeline(
            lookback_days="2 days ago UTC",
            threshold_percent=0.5,
            lookahead_periods=6,
            save_models=True
        )
        
        # Reload models in prediction API
        reload_response = requests.post(f"{PREDICTION_API_URL}/model/reload")
        
        return {
            'status': 'success',
            'message': 'Retraining completed',
            'model_dir': results['model_dir'],
            'f1_score': results['ensemble_f1']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(e)}"
        )


@app.get('/model/info')
def model_info():
    """Get model information from prediction API"""
    try:
        response = requests.get(f"{PREDICTION_API_URL}/model/info")
        return response.json()
    except:
        return {
            'error': 'Could not fetch model info',
            'prediction_api': PREDICTION_API_URL
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
