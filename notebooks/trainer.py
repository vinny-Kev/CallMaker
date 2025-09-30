import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
from feature_engineer import generate_all_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import joblib

# SECURITY: Load API keys from .env
load_dotenv()
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

assert API_KEY and API_SECRET, "Binance API keys must be set in .env!"

client = Client(API_KEY, API_SECRET)

# Fetch historical 5m klines for last N days (batched)
def fetch_historical_klines(symbol, interval, days=3):
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    klines = []
    while start_time < end_time:
        batch = client.get_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=min(end_time, start_time + 1000*60*60*24))
        if not batch:
            break
        klines.extend(batch)
        start_time = batch[-1][0] + 1
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df

# Feature engineering stub
def make_features(df):
    return generate_all_features(df)

# Labeling stub
def make_labels(df):
    # TODO: Implement binary labeling logic
    df['label'] = 0
    return df

# Model stubs
def train_models(X, y):
    models = {}
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    models['rf'] = rf
    # CatBoost
    cb = CatBoostClassifier(iterations=100, depth=5, l2_leaf_reg=3, verbose=0, random_seed=42)
    cb.fit(X, y)
    models['catboost'] = cb
    # Logistic Regression
    lr = LogisticRegression(penalty='l2', class_weight='balanced', solver='liblinear', random_state=42)
    lr.fit(X, y)
    models['lr'] = lr
    # LSTM stub (fit on dummy data)
    # TODO: Implement sequence data prep
    lstm_model = Sequential([
        LSTM(16, input_shape=(10, X.shape[1]), dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # lstm_model.fit(...)
    models['lstm'] = lstm_model
    return models

# Ensemble stub
def ensemble_predict(models, X):
    # TODO: Implement weighted soft voting
    preds = np.mean([
        models['rf'].predict_proba(X)[:,1],
        models['catboost'].predict_proba(X)[:,1],
        models['lr'].predict_proba(X)[:,1]
    ], axis=0)
    return preds

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = '5m'
    df = fetch_historical_klines(symbol, interval, days=3)
    df = make_features(df)
    df = make_labels(df)
    X = df.drop(['label'], axis=1)
    y = df['label']
    models = train_models(X, y)
    # Save models
    for name, model in models.items():
        if name == 'lstm':
            model.save(f'data/models/{name}.h5')
        else:
            joblib.dump(model, f'data/models/{name}.pkl')
    # Save ensemble weights stub
    joblib.dump({'rf':1/3, 'catboost':1/3, 'lr':1/3}, 'data/models/ensemble_weights.pkl')
