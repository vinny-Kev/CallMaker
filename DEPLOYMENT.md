# 🚀 Model Deployment Workflow

## Quick Start

After training a new model, simply run:

### Option 1: Python Script (Recommended)
```bash
python deploy_model_to_api.py
```

### Option 2: PowerShell Script (Even Faster!)
```powershell
.\deploy.ps1
```

## What It Does

The deployment script automatically:

1. ✅ Finds the latest trained model in `data/models/`
2. ✅ Displays model information and performance metrics
3. ✅ Removes old models from the API directory
4. ✅ Copies the new model to `d:\CODE ALL HERE PLEASE\bitcoin-api-service\data\models\`
5. ✅ Shows deployment summary with model details

## Complete Training + Deployment Workflow

```bash
# Step 1: Train a new model
python -m src.train_pipeline

# Step 2: Deploy to API
python deploy_model_to_api.py

# Step 3: Restart API service (on Render, it auto-deploys on git push)
```

## Example Output

```
================================================================================
🚀 MODEL DEPLOYMENT TO API SERVICE
================================================================================

📦 Latest Model Found: BTCUSDT_1m_20251006_161058

📊 Model Information:
  Symbol: BTCUSDT
  Interval: 1m
  Features: 20
  Architecture: 3-model ensemble: CatBoost + Random Forest + Logistic Regression
  Training Date: 20251006_161058

📈 Performance Metrics:
  Train Accuracy: 96.46%
  Test Accuracy:  84.79%
  Test F1 Score:  0.3084
  Test ROC AUC:   0.6569

✅ DEPLOYMENT SUCCESSFUL!
```

## Notes

- The script always deploys the **most recent** model (based on timestamp in folder name)
- Old models in the API directory are automatically removed to save space
- Model metadata includes comprehensive metrics (accuracy, precision, recall, F1, ROC AUC)
- The API service needs to be restarted to load the new model
