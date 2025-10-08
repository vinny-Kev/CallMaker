"""
Quick Model Save Script
Just save the model that was just trained
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models_transformer_regression_v2 import TransformerRegressorV2

def create_and_save_model():
    """Create a model instance and save it properly"""
    
    print("Creating model for saving...")
    
    # Model config (same as training)
    model = TransformerRegressorV2(
        sequence_length=60,
        n_features=106,
        d_model=64,
        num_heads=4,
        num_blocks=2,
        ff_dim=128,
        dropout=0.1,
        loss_type='mae'
    )
    
    # Build the model
    model.build()
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f"data/models/BTCUSDT_4h_mae_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save using the proper method
    print(f"Saving to {model_dir}...")
    model.save_model(model_dir)
    
    print(f"✅ Model saved successfully to {model_dir}")
    return model_dir

if __name__ == '__main__':
    create_and_save_model()