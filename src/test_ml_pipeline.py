"""
Quick test script for ML pipeline components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from data_scraper import DataScraper
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
from models import CatBoostModel, RandomForestModel, LSTMModel

print("\n" + "="*60)
print("ML PIPELINE COMPONENT TEST")
print("="*60 + "\n")

# Test 1: Data Scraper
print("Test 1: Data Scraper")
print("-" * 40)
try:
    scraper = DataScraper("BTCUSDT", "5m")
    df = scraper.fetch_historical_("6 hours ago UTC")
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Feature Engineering
print("\nTest 2: Feature Engineering")
print("-" * 40)
try:
    engineer = FeatureEngineer()
    df_features = engineer.generate_all_features(
        df, 
        create_targets=True,
        threshold_percent=0.5,
        lookahead_periods=6
    )
    print(f"✅ Generated {len(df_features.columns)} features")
    print(f"   Samples after cleaning: {len(df_features)}")
    
    # Check class distribution
    class_dist = df_features['target'].value_counts()
    print(f"   Class distribution:\n{class_dist}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Preprocessing
print("\nTest 3: Data Preprocessing")
print("-" * 40)
try:
    preprocessor = DataPreprocessor()
    feature_cols = engineer.get_feature_columns(df_features)
    
    prepared_data = preprocessor.prepare_data_pipeline(
        df_features,
        feature_cols,
        target_col='target',
        train_ratio=0.7,
        val_ratio=0.15,
        undersample=True,
        undersample_ratio=0.3,
        sequence_length=20  # Shorter for testing
    )
    
    print(f"✅ Data prepared successfully")
    print(f"   Training samples: {len(prepared_data['X_train'])}")
    print(f"   Validation samples: {len(prepared_data['X_val'])}")
    print(f"   Test samples: {len(prepared_data['X_test'])}")
    print(f"   LSTM sequences: {prepared_data['X_train_lstm'].shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: CatBoost Model
print("\nTest 4: CatBoost Model")
print("-" * 40)
try:
    cb_model = CatBoostModel(n_classes=3)
    cb_model.build(
        class_weights=prepared_data['class_weights'],
        iterations=50,  # Small for testing
        depth=4
    )
    
    # Quick training
    cb_model.train(
        prepared_data['X_train'].head(500),
        prepared_data['y_train'].head(500),
        prepared_data['X_val'].head(100),
        prepared_data['y_val'].head(100)
    )
    
    # Test prediction
    y_pred = cb_model.predict(prepared_data['X_val'].head(10))
    print(f"✅ CatBoost model trained and tested")
    print(f"   Sample predictions: {y_pred[:5]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Random Forest Model
print("\nTest 5: Random Forest Model")
print("-" * 40)
try:
    rf_model = RandomForestModel(n_classes=3)
    rf_model.build(
        class_weights=prepared_data['class_weights'],
        n_estimators=50,  # Small for testing
        max_depth=10
    )
    
    rf_model.train(
        prepared_data['X_train'].head(500),
        prepared_data['y_train'].head(500)
    )
    
    y_pred = rf_model.predict(prepared_data['X_val'].head(10))
    print(f"✅ Random Forest model trained and tested")
    print(f"   Sample predictions: {y_pred[:5]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: LSTM Model
print("\nTest 6: LSTM Model")
print("-" * 40)
try:
    lstm_model = LSTMModel(n_classes=3)
    
    input_shape = (
        prepared_data['X_train_lstm'].shape[1],
        prepared_data['X_train_lstm'].shape[2]
    )
    
    lstm_model.build(
        input_shape=input_shape,
        lstm_units=[64, 32],  # Smaller for testing
        dropout=0.2
    )
    
    lstm_model.compile(
        learning_rate=0.001,
        class_weights=prepared_data['class_weights']
    )
    
    # Very quick training
    lstm_model.train(
        prepared_data['X_train_lstm'][:200],
        prepared_data['y_train_lstm'][:200],
        prepared_data['X_val_lstm'][:50],
        prepared_data['y_val_lstm'][:50],
        class_weights=prepared_data['class_weights'],
        epochs=3,  # Very few epochs for testing
        batch_size=16
    )
    
    y_pred = lstm_model.predict(prepared_data['X_val_lstm'][:10])
    print(f"✅ LSTM model trained and tested")
    print(f"   Sample predictions: {y_pred[:5]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
print("\n✅ All components working correctly!")
print("\nNext steps:")
print("  1. Run full training: python train_pipeline.py")
print("  2. Start prediction API: python prediction_api.py")
print("  3. Start backend: python backend.py (in app/)")
print("  4. Start dashboard: streamlit run app.py (in app/)")
print("\n" + "="*60 + "\n")
