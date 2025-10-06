# 🎯 Bitcoin AI Prediction System - Training Pipeline# 🚀 Bitcoin AI Trading Dashboard# Money Printer — Trading Assistant (MVP)



**ML training pipeline for the Bitcoin price prediction system**



## 🚀 Overview> Real-time Bitcoin analysis with AI-powered price movement predictions**What it is**



This repository contains the machine learning training pipeline that powers the Bitcoin AI Prediction API. It uses an optimized 3-model ensemble that achieved **91.09% accuracy** after removing the unnecessary LSTM complexity.A research-first trading assistant (non-autonomous) that uses a heterogeneous ensemble to produce high-conviction signals and visualizes them on live charts. This repo is a remaster of a previous "money printer" experiment — built to be reproducible, auditable, and safe for demos.



## 🏗️ Architecture## 🎯 What This Does



### Optimized 3-Model Ensemble**Stack**

- **CatBoost (50%)** - Gradient boosting excellence

- **Random Forest (25%)** - Feature importance & generalization  - **Real-time Bitcoin charts** with candlestick data- Frontend / demo: Streamlit

- **Logistic Regression (25%)** - Linear baseline & fast inference

- **AI predictions** using ensemble ML models (CatBoost + Random Forest + LSTM)- Models: CatBoost, RandomForest, LogisticRegression, LSTM (Keras)

### Why No LSTM?

✅ **Better Performance**: 91.09% vs 89.21% (+1.88%)  - **Price movement forecasts** with confidence scores- Storage: parquet / joblib (.pkl), optional sqlite

✅ **5x Faster**: Training and inference  

✅ **More Reliable**: No TensorFlow compatibility issues  - **Orange prediction overlay** on charts showing future price path- Data source: Binance (REST + websockets)

✅ **Simpler**: Easier deployment and maintenance  

- **BUY/SELL/HOLD signals** with probability breakdowns- Language: Python 3.9+

## 📊 Performance



| Metric | Value |

|--------|-------|---**Key constraints**

| **Test Accuracy** | 91.09% |

| **F1 Score** | 0.3182 |- NO SMOTE (no synthetic oversampling).

| **Training Time** | ~1 minute |

| **Features** | 70 engineered features |## ⚡ Quick Start (3 Steps)- Retrain daily (scriptable via cron / GitHub Actions).



## 🛠️ Setup- No automatic trading. UI must clearly state “research-only — no trading enabled.”



### Prerequisites### 1. Install Dependencies- Use class_weight or BalancedBagging for imbalance; prefer cost-sensitive methods over resampling.

- Python 3.11+

- Binance API keys```powershell- Soft-voting ensemble with weights derived from out-of-sample performance and stability.



### Installation# Install Python packages

```bash

git clone <this-repo>pip install -r requirements.txt---

cd money-printer

pip install -r requirements.txt```

```

## Repo layout

### Environment Setup

```bash### 2. Set Up API Keysmoney-printer/

cp .env.example .env

# Add your Binance API keys:Create a `.env` file in the root directory:├─ data/

# BINANCE_API_KEY=your_api_key

# BINANCE_SECRET_KEY=your_secret_key```├─ src/

```

BINANCE_API_KEY=your_binance_api_key_here│ ├─ fetcher.py

## 🚀 Usage

BINANCE_SECRET_KEY=your_binance_secret_key_here│ ├─ fe.py

### Quick Training

```bash```│ ├─ label.py

python src/train_pipeline.py

```│ ├─ train.py



### Training Pipeline Steps### 3. Run the System│ ├─ ensemble.py

1. **Data Collection** - Fetch BTCUSDT 1m data from Binance

2. **Feature Engineering** - 70 technical indicators```powershell│ ├─ backtest.py

3. **Data Preprocessing** - Scaling, temporal splits, class balancing

4. **Model Training** - Train 3-model ensemble# Terminal 1: Start ML Prediction API│ └─ utils.py

5. **Evaluation** - Performance metrics and validation

6. **Model Saving** - Export for deploymentcd src├─ app/



### Key Componentspython prediction_api.py│ └─ streamlit_app.py



#### `src/train_pipeline.py`├─ notebooks/

Main training orchestrator

# Terminal 2: Start Backend API  ├─ requirements.txt

#### `src/feature_engineering.py`

70+ technical indicators:cd app├─ README.md

- Price features (SMA, EMA, returns)

- Technical indicators (RSI, MACD, Bollinger Bands)python backend.py└─ .env.example

- Volume indicators (VWAP, volume MA)

- Volatility measures (ATR, volatility)

- Time-based features

# Terminal 3: Start Dashboard---

#### `src/models.py`

Clean 3-model ensemble implementation:streamlit run app/app.py

- CatBoostModel

- RandomForestModel  ```## Quickstart (development)

- LogisticRegressionModel

- EnsembleModel1. Clone:



#### `src/preprocessing.py`**That's it!** Open http://localhost:8501 in your browser.```bash

Data preprocessing pipeline:

- Robust scalinggit clone https://github.com/vinny-Kev/CallMaker.git

- Temporal data splits

- Class weight balancing---cd money-printer ## or the name of the directory the project is cloned in

- Feature selection

python -m venv .venv

## 📁 Directory Structure

## 🎮 How to Usesource .venv/bin/activate     # linux / mac

```

money-printer/.venv\Scripts\activate        # windows

├── src/

│   ├── train_pipeline.py      # Main training script1. **Dashboard opens** → Bitcoin data loads automaticallypip install -r requirements.txt

│   ├── models.py              # 3-model ensemble

│   ├── feature_engineering.py # Technical indicators2. **Click "🤖 Get ML Prediction"** → See AI forecast with orange overlay```

│   ├── preprocessing.py       # Data preprocessing

│   ├── data_scraper.py        # Binance data fetching3. **View results:**Copy environment example:

│   └── prediction_api.py      # API interface

├── data/   - BUY 🟢 / SELL 🔴 / HOLD ⚪ signal```bash

│   ├── models/               # Trained model outputs

│   ├── processed/           # Processed datasets   - Confidence percentagecp .env.example .env

│   └── raw/                 # Raw market data

├── notebooks/               # Jupyter notebooks (optional)   - Price forecast for next 6 periods# Edit .env to add BINANCE_API_KEY and BINANCE_API_SECRET (read-only ideally)

├── requirements.txt         # Python dependencies

└── .env                    # Environment variables   - Probability breakdown```

```

Fetch historical data (example):

## 🎯 Model Output

---```bash

Trained models are saved to `data/models/BTCUSDT_1m_YYYYMMDD_HHMMSS/`:

- `catboost_model.cbm` - CatBoost modelpython -m src.fetcher fetch_historical --symbol BTCUSDT --interval 1m --limit 1000

- `random_forest_model.pkl` - Random Forest model  

- `logistic_model.pkl` - Logistic Regression model## 🔧 System Architecture```

- `ensemble_weights.json` - Model weights

- `preprocessor.pkl` - Feature scalerTrain small quick models (MVP):

- `feature_columns.pkl` - Feature names

- `metadata.json` - Training metadata``````bash



## 🔄 Model DeploymentBitcoin Data (Binance API)python -m src.train --symbol BTCUSDT --interval 1m --limit 2000 --save-dir data/models



After training, models are deployed to:    ↓```

1. **API Service**: [BTC-Forecast-API](https://github.com/vinny-Kev/BTC-Forecast-API)

2. **Streamlit Demo**: [bitcoin-prediction-demo](https://github.com/your-username/bitcoin-streamlit-demo)Feature Engineering (80+ indicators)Run Streamlit demo:



## 📈 Feature Engineering    ↓```bash



### Technical Indicators (70 features)ML Ensemble Modelsstreamlit run app/streamlit_app.py

- **Trend**: SMA, EMA, MACD, ADX

- **Momentum**: RSI, ROC, Williams %R    ↓```

- **Volatility**: ATR, Bollinger Bands, volatility measures

- **Volume**: Volume MA, VWAP, volume indicatorsPrediction API (port 8000)

- **Price**: Returns, highs/lows, price ratios

- **Time**: Hour, day, week patterns    ↓### Configuration



### Target ClassesBackend API (port 8001)Main configuration lives in src/utils.py or as CLI args. Important parameters:

- **0**: No Significant Movement (< 0.5% change)

- **1**: Large Upward Movement (> 0.5% up)      ↓- CONTEXT_WINDOW (default: 120)

- **2**: Large Downward Movement (> 0.5% down)

Streamlit Dashboard (port 8501)- HORIZON (label horizon) (default: 30)

## 🧪 Hyperparameters

```- LABEL_THRESHOLD (example: 0.01 for 1% move)

### Optimized Settings

- **CatBoost**: 500 iterations, depth 6, learning_rate 0.1- DAILY_RETRAIN_SCHEDULE (cron or GitHub Action recommended)

- **Random Forest**: 50 trees, max_depth 8 (reduced complexity)

- **Logistic Regression**: C=1.0, lbfgs solver---- TRANSACTION_COST (for backtests)

- **Ensemble Weights**: [0.5, 0.25, 0.25]



## 📊 Performance History

## 📊 What the AI Predicts### Design decisions & noise handling

| Version | Accuracy | F1 Score | Notes |

|---------|----------|----------|--------|- Heterogeneous ensemble reduces correlated errors.

| v1.0 (4-model) | 89.21% | 0.3147 | With LSTM |

| **v2.0 (3-model)** | **91.09%** | **0.3182** | **LSTM removed** |- **Class 0**: No significant movement (±0.5%)- Soft voting on predicted probabilities (not hard class labels).



## 🔧 Customization- **Class 1**: Large upward movement (≥+0.5%)- Weights derived from out-of-sample AUC/AUC-PR and fold-level prediction volatility. Low-volatility stable models get higher weight.



### Modify Features- **Class 2**: Large downward movement (≤-0.5%)- Per-model noise mitigation: RF (tune min_samples_leaf, n_estimators); CatBoost (L2 reg, cat feature handling); LSTM (dropout, early stopping); LR (robust scaling + regularization).

Edit `src/feature_engineering.py` to add/remove technical indicators

- Preprocessing: moving averages, RSI, volatility (rolling std), momentum, and optional Savitzky-Golay / wavelet smoothing.

### Adjust Models

Modify `src/models.py` to tune hyperparameters or add models**Models Used:**- Imbalance: class weights or BalancedBaggingClassifier. No SMOTE.



### Change Data Source- **CatBoost** (40% weight) - Gradient boosting

Update `src/data_scraper.py` for different symbols/timeframes

- **Random Forest** (30% weight) - Tree ensemble  ### Acceptance criteria

## 📝 Contributing

- **LSTM** (30% weight) - Deep learning- fetch_historical saves data/raw/<symbol>_<interval>.parquet.

1. Fork the repository

2. Create feature branch- train.py produces working artifacts in data/models/ (RF/CB/LR pkl and LSTM .h5).

3. Make changes

4. Test thoroughly  ---- Streamlit app loads latest model artifacts and overlays markers for ensemble_proba > user_threshold.

5. Submit pull request

- Backtest prints AUC-PR + precision@k and saves results csv.

## 📄 License

## 🛠️ Training New Models (Optional)

MIT License

### Security & Ops

---

```powershell- Put BINANCE_API_KEY and BINANCE_API_SECRET in .env and use python-dotenv.

**🚀 This training pipeline powers the live Bitcoin AI Prediction API with 91.09% accuracy!**
cd src- Use read-only API key for dev; never commit keys.

python train_pipeline.py- Use exponential backoff and reconnect logic in websocket code.

```- Store model artifacts with meta.json documenting training range, seed, and metrics.



This will:### Roadmap / Next steps

- Fetch 2 days of Bitcoin data- SHAP-based explainability per marker (RF & CatBoost).

- Generate 79 technical indicators- LSTM improvements & sequence attention.

- Train all 3 models- GitHub Actions to run scheduled daily retrain (requires secrets).

- Save to `data/models/`- Optional: TimeGAN/CTGAN augmentation for extreme rare-class cases (research only).



**Training takes 10-20 minutes.**### License

MIT

---

### Contact

## 🚨 TroubleshootingCreator: Kevin — use repo issues / PRs for feedback.


### APIs Not Starting?
```powershell
# Kill any existing Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Then restart the 3 commands above
```

### Models Not Found?
```powershell
# Train new models
cd src
python train_pipeline.py
```

### Dashboard Shows "APIs Offline"?
1. Check that all 3 terminals are running
2. Click "🔍 Check API Status" in sidebar
3. Look for these URLs:
   - http://localhost:8000 (Prediction API)
   - http://localhost:8001 (Backend API)
   - http://localhost:8501 (Dashboard)

### Connection Errors?
- Check your `.env` file has valid Binance API keys
- Click "🔄 Refresh Data" in the sidebar

---

## 📁 Project Structure

```
money-printer/
├── app/
│   ├── app.py          # 🎯 Main dashboard
│   └── backend.py      # API proxy
├── src/
│   ├── prediction_api.py    # 🤖 ML prediction service
│   ├── train_pipeline.py    # 🎓 Model training
│   ├── data_scraper.py      # 📊 Binance data
│   ├── feature_engineering.py  # ⚙️ Technical indicators
│   ├── preprocessing.py     # 🔧 Data preparation
│   └── models.py           # 🧠 ML models
├── data/models/        # 💾 Trained models
├── requirements.txt    # 📋 Dependencies
└── .env               # 🔐 API keys
```

---

## ⚠️ Important Notes

- **Research tool only** - Not for actual trading
- **Educational purposes** - Not financial advice
- **Models trained on 2 days** - Limited historical data
- **Predictions not guaranteed** - Past performance ≠ future results

---

## 🎯 Current Model Performance

- **Overall Accuracy**: 81%
- **Best at predicting**: No movement (89% F1 score)
- **Struggles with**: Large up/down movements (0% recall)
- **Recommendation**: Retrain with 30+ days of data

**See `MODEL_PERFORMANCE_REPORT.md` for detailed analysis.**

---

## 🚀 That's It!

Run the 3 commands, open the dashboard, and start seeing AI predictions on Bitcoin charts!

**Questions?** The dashboard shows helpful error messages and instructions in the sidebar.