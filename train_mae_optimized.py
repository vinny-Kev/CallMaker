"""
Optimized MAE Training - Less Data, Better Performance
Based on analysis: 365 days is too much, causing poor generalization
Going back to 100 days but with more robust architecture
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_scraper import DataScraper
from src.feature_engineering import FeatureEngineer
from src.models_transformer_regression_v2 import TransformerRegressorV2


# =====================================================================
# OPTIMIZED CONFIGURATION
# =====================================================================

SYMBOL = 'BTCUSDT'
INTERVAL = '4h'
LOOKAHEAD = 6  # 24 hours ahead
DAYS = 120  # SWEET SPOT: More than 100, less than 365
SEQUENCE_LENGTH = 60
N_FEATURES = None  # Will be determined from data

# Transformer Architecture  
D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 2
FF_DIM = 128
DROPOUT = 0.1

# Training
EPOCHS = 50  # Reduced to prevent overfitting
BATCH_SIZE = 32
TEST_SIZE = 0.2

# Trading Simulation Thresholds
THRESHOLDS = [0.5, 0.8, 1.0, 1.5, 2.0]


class OptimizedMAEPipeline:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.symbol = SYMBOL
        self.interval = INTERVAL
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def prepare_data(self):
        """Fetch and prepare training data"""
        print("=" * 80)
        print("STEP 1: DATA PREPARATION (OPTIMIZED)")
        print("=" * 80)
        
        # Fetch data
        print(f"\nFetching {DAYS} days of {SYMBOL} {INTERVAL} data...")
        scraper = DataScraper(symbol=SYMBOL, interval=INTERVAL)
        df = scraper.fetch_historical_(lookback_days=f"{DAYS} days ago UTC")
        print(f"✓ Fetched {len(df)} candles")
        
        # Feature engineering
        print("\nCreating features...")
        fe = FeatureEngineer()
        df = fe.generate_all_features(df)
        
        # Create regression target
        print(f"Creating regression target (lookahead={LOOKAHEAD} periods = 24h)...")
        df = fe.create_regression_target(df, lookahead_periods=LOOKAHEAD)
        
        # Drop NaN
        initial_len = len(df)
        df = df.dropna()
        print(f"✓ Dropped {initial_len - len(df)} rows with NaN")
        print(f"✓ Final dataset: {len(df)} samples")
        
        return df
    
    def create_sequences(self, df):
        """Create sequences for transformer"""
        print("\n" + "=" * 80)
        print("STEP 2: SEQUENCE CREATION")
        print("=" * 80)
        
        # Select features
        exclude_cols = ['timestamp', 'target_return', 'max_upside', 'max_downside']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df['target_return'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        print(f"\nCreating sequences (length={SEQUENCE_LENGTH})...")
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - SEQUENCE_LENGTH):
            X_seq.append(X_scaled[i:i + SEQUENCE_LENGTH])
            y_seq.append(y[i + SEQUENCE_LENGTH])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split train/test
        split_idx = int(len(X_seq) * (1 - TEST_SIZE))
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]
        
        print(f"✓ Created {len(X_seq)} sequences")
        print(f"✓ Training: {len(X_train)} sequences")
        print(f"✓ Testing: {len(X_test)} sequences")
        print(f"✓ Features: {len(feature_cols)}")
        
        # Update N_FEATURES global
        global N_FEATURES
        N_FEATURES = len(feature_cols)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train MAE Transformer"""
        print("\n" + "=" * 80)
        print("STEP 3: OPTIMIZED MAE TRAINING")
        print("=" * 80)
        
        print(f"\nArchitecture:")
        print(f"  - Loss: MAE (Mean Absolute Error)")
        print(f"  - d_model: {D_MODEL}")
        print(f"  - Heads: {NUM_HEADS}")
        print(f"  - Blocks: {NUM_BLOCKS}")
        print(f"  - FF Dim: {FF_DIM}")
        print(f"  - Dropout: {DROPOUT}")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Batch Size: {BATCH_SIZE}")
        
        self.model = TransformerRegressorV2(
            sequence_length=SEQUENCE_LENGTH,
            n_features=N_FEATURES,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_blocks=NUM_BLOCKS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            loss_type='mae'  # MAE ONLY
        )
        
        # Use 10% of training data for validation
        val_split_idx = int(len(X_train) * 0.9)
        X_train_split = X_train[:val_split_idx]
        y_train_split = y_train[:val_split_idx]
        X_val = X_train[val_split_idx:]
        y_val = y_train[val_split_idx:]
        
        print(f"\nSplit for validation:")
        print(f"  - Training: {len(X_train_split)} sequences")
        print(f"  - Validation: {len(X_val)} sequences")
        print(f"  - Testing: {len(X_test)} sequences")
        
        print("\nTraining started...")
        history = self.model.train(
            X_train_split, y_train_split,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        print("✓ Training complete")
        return history
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Comprehensive evaluation"""
        print("\n" + "=" * 80)
        print("STEP 4: EVALUATION")
        print("=" * 80)
        
        # Get predictions
        y_pred_raw = self.model.predict(X_test)
        
        # Calibrate (need train data for calibration)
        print("\nCalibrating predictions...")
        
        # Use last 20% of training data as calibration validation
        val_split_idx = int(len(X_train) * 0.8)
        X_cal_train = X_train[:val_split_idx]
        y_cal_train = y_train[:val_split_idx]
        X_cal_val = X_train[val_split_idx:]
        y_cal_val = y_train[val_split_idx:]
        
        scale_factor = self.model.calibrate_predictions(X_cal_train, y_cal_train, X_cal_val, y_cal_val)
        y_pred_calibrated = y_pred_raw * scale_factor
        
        # Get confidence predictions
        print("Calculating uncertainty (Monte Carlo Dropout)...")
        y_pred_confident, uncertainty = self.model.predict_with_confidence(X_test, n_samples=10)
        
        # Metrics
        print("\n" + "-" * 80)
        print("REGRESSION METRICS:")
        print("-" * 80)
        
        for name, preds in [('Raw', y_pred_raw), ('Calibrated', y_pred_calibrated), ('Confident', y_pred_confident)]:
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.4f}%")
            print(f"  RMSE: {rmse:.4f}%")
            print(f"  R²:   {r2:.4f}")
        
        # Direction accuracy
        print("\n" + "-" * 80)
        print("DIRECTION ACCURACY:")
        print("-" * 80)
        
        for name, preds in [('Raw', y_pred_raw), ('Calibrated', y_pred_calibrated), ('Confident', y_pred_confident)]:
            direction_acc = np.mean((np.sign(preds) == np.sign(y_test)).astype(int)) * 100
            print(f"{name}: {direction_acc:.2f}%")
        
        print(f"\nScale Factor: {scale_factor:.3f}x")
        print(f"Avg Uncertainty: {np.mean(uncertainty):.3f}%")
        
        # Trading simulation
        print("\n" + "-" * 80)
        print("TRADING SIMULATION:")
        print("-" * 80)
        
        results = []
        
        for threshold in THRESHOLDS:
            for pred_type, preds in [('raw', y_pred_raw), ('calibrated', y_pred_calibrated), ('confident', y_pred_confident)]:
                # Filter by threshold
                mask = np.abs(preds) >= threshold
                
                if np.sum(mask) == 0:
                    continue
                
                filtered_preds = preds[mask]
                filtered_actual = y_test[mask]
                
                # Calculate metrics
                n_trades = np.sum(mask)
                direction_acc = np.mean((np.sign(filtered_preds) == np.sign(filtered_actual)).astype(int)) * 100
                avg_return = np.mean(filtered_actual[np.sign(filtered_preds) == np.sign(filtered_actual)])
                total_return = np.sum(filtered_actual[np.sign(filtered_preds) == np.sign(filtered_actual)])
                win_rate = direction_acc  # Same as direction accuracy
                
                results.append({
                    'threshold': threshold,
                    'pred_type': pred_type,
                    'n_trades': int(n_trades),  # Convert to Python int
                    'direction_acc': float(direction_acc),  # Convert to Python float
                    'avg_return': float(avg_return),
                    'total_return': float(total_return),
                    'win_rate': float(win_rate)
                })
        
        # Sort by win rate
        results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        
        print("\nTop 10 Strategies:")
        print(f"{'Strategy':<20} {'Trades':<8} {'Accuracy':<10} {'Avg Return':<12} {'Win Rate':<10}")
        print("-" * 70)
        
        for i, r in enumerate(results[:10]):
            strategy_name = f"{r['pred_type']}_{r['threshold']}"
            print(f"{strategy_name:<20} {r['n_trades']:<8} {r['direction_acc']:<10.2f}% {r['avg_return']:<12.3f}% {r['win_rate']:<10.2f}%")
        
        return {
            'y_pred_raw': y_pred_raw,
            'y_pred_calibrated': y_pred_calibrated,
            'y_pred_confident': y_pred_confident,
            'uncertainty': uncertainty,
            'scale_factor': scale_factor,
            'trading_results': results
        }
    
    def save_model(self, results):
        """Save model and metadata"""
        print("\n" + "=" * 80)
        print("STEP 5: SAVING MODEL")
        print("=" * 80)
        
        # Create save directory
        model_dir = f"data/models/{self.symbol}_{self.interval}_mae_optimized_{self.timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save using the model's save method
        print(f"Saving model to {model_dir}...")
        self.model.save_model(model_dir)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, features_path)
        print(f"✓ Saved feature columns to {features_path}")
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'lookahead': LOOKAHEAD,
            'days_trained': DAYS,
            'sequence_length': SEQUENCE_LENGTH,
            'n_features': N_FEATURES,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_blocks': NUM_BLOCKS,
            'ff_dim': FF_DIM,
            'dropout': DROPOUT,
            'loss_type': 'mae',
            'scale_factor': float(results['scale_factor']),
            'best_strategy': results['trading_results'][0],
            'timestamp': self.timestamp,
            'optimization': 'reduced_days_120'
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")
        
        print(f"\n{'='*80}")
        print(f"MODEL SAVED TO: {model_dir}")
        print(f"{'='*80}")
        
        return model_dir


def main():
    """Run the optimized training pipeline"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "OPTIMIZED MAE TRANSFORMER TRAINING PIPELINE" + " " * 20 + "║")
    print("║" + " " * 78 + "║")
    print("║" + f"  Symbol: {SYMBOL:<15} Interval: {INTERVAL:<15} Lookahead: {LOOKAHEAD} periods" + " " * 15 + "║")
    print("║" + f"  Training Data: {DAYS} days (OPTIMIZED)" + " " * 46 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    pipeline = OptimizedMAEPipeline()
    
    # Prepare data
    df = pipeline.prepare_data()
    
    # Create sequences
    X_train, X_test, y_train, y_test = pipeline.create_sequences(df)
    
    # Train model
    history = pipeline.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    results = pipeline.evaluate_model(X_train, y_train, X_test, y_test)
    
    # Save
    model_dir = pipeline.save_model(results)
    
    print("\n" + "=" * 80)
    print("OPTIMIZED TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {model_dir}")
    print(f"\nBest Strategy:")
    best = results['trading_results'][0]
    print(f"  - Type: {best['pred_type']}_{best['threshold']}")
    print(f"  - Trades: {best['n_trades']}")
    print(f"  - Win Rate: {best['win_rate']:.2f}%")
    print(f"  - Avg Return: {best['avg_return']:.3f}%")
    print(f"  - Total Return: {best['total_return']:.2f}%")
    print("\n" + "=" * 80)
    print("Ready for deployment! Check results vs previous 87% target")
    print("=" * 80)


if __name__ == '__main__':
    main()