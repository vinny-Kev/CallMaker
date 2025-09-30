# Money Printer — Trading Assistant (MVP)

**What it is**
A research-first trading assistant (non-autonomous) that uses a heterogeneous ensemble to produce high-conviction signals and visualizes them on live charts. This repo is a remaster of a previous "money printer" experiment — built to be reproducible, auditable, and safe for demos.

**Stack**
- Frontend / demo: Streamlit
- Models: CatBoost, RandomForest, LogisticRegression, LSTM (Keras)
- Storage: parquet / joblib (.pkl), optional sqlite
- Data source: Binance (REST + websockets)
- Language: Python 3.9+

**Key constraints**
- NO SMOTE (no synthetic oversampling).
- Retrain daily (scriptable via cron / GitHub Actions).
- No automatic trading. UI must clearly state “research-only — no trading enabled.”
- Use class_weight or BalancedBagging for imbalance; prefer cost-sensitive methods over resampling.
- Soft-voting ensemble with weights derived from out-of-sample performance and stability.

---

## Repo layout
money-printer/
├─ data/
├─ src/
│ ├─ fetcher.py
│ ├─ fe.py
│ ├─ label.py
│ ├─ train.py
│ ├─ ensemble.py
│ ├─ backtest.py
│ └─ utils.py
├─ app/
│ └─ streamlit_app.py
├─ notebooks/
├─ requirements.txt
├─ README.md
└─ .env.example

---

## Quickstart (development)
1. Clone:
```bash
git clone <REPO_URL>
cd money-printer
python -m venv .venv
source .venv/bin/activate     # linux / mac
.venv\Scripts\activate        # windows
pip install -r requirements.txt
```
Copy environment example:
```bash
cp .env.example .env
# Edit .env to add BINANCE_API_KEY and BINANCE_API_SECRET (read-only ideally)
```
Fetch historical data (example):
```bash
python -m src.fetcher fetch_historical --symbol BTCUSDT --interval 1m --limit 1000
```
Train small quick models (MVP):
```bash
python -m src.train --symbol BTCUSDT --interval 1m --limit 2000 --save-dir data/models
```
Run Streamlit demo:
```bash
streamlit run app/streamlit_app.py
```

### Configuration
Main configuration lives in src/utils.py or as CLI args. Important parameters:
- CONTEXT_WINDOW (default: 120)
- HORIZON (label horizon) (default: 30)
- LABEL_THRESHOLD (example: 0.01 for 1% move)
- DAILY_RETRAIN_SCHEDULE (cron or GitHub Action recommended)
- TRANSACTION_COST (for backtests)

### Design decisions & noise handling
- Heterogeneous ensemble reduces correlated errors.
- Soft voting on predicted probabilities (not hard class labels).
- Weights derived from out-of-sample AUC/AUC-PR and fold-level prediction volatility. Low-volatility stable models get higher weight.
- Per-model noise mitigation: RF (tune min_samples_leaf, n_estimators); CatBoost (L2 reg, cat feature handling); LSTM (dropout, early stopping); LR (robust scaling + regularization).
- Preprocessing: moving averages, RSI, volatility (rolling std), momentum, and optional Savitzky-Golay / wavelet smoothing.
- Imbalance: class weights or BalancedBaggingClassifier. No SMOTE.

### Acceptance criteria
- fetch_historical saves data/raw/<symbol>_<interval>.parquet.
- train.py produces working artifacts in data/models/ (RF/CB/LR pkl and LSTM .h5).
- Streamlit app loads latest model artifacts and overlays markers for ensemble_proba > user_threshold.
- Backtest prints AUC-PR + precision@k and saves results csv.

### Security & Ops
- Put BINANCE_API_KEY and BINANCE_API_SECRET in .env and use python-dotenv.
- Use read-only API key for dev; never commit keys.
- Use exponential backoff and reconnect logic in websocket code.
- Store model artifacts with meta.json documenting training range, seed, and metrics.

### Roadmap / Next steps
- SHAP-based explainability per marker (RF & CatBoost).
- LSTM improvements & sequence attention.
- GitHub Actions to run scheduled daily retrain (requires secrets).
- Optional: TimeGAN/CTGAN augmentation for extreme rare-class cases (research only).

### License
MIT

### Contact
Creator: Kevin — use repo issues / PRs for feedback.
