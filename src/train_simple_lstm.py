"""
Simple LSTM Training Pipeline - NO META-LEARNER
Clean, direct LSTM predictions without ensemble complexity

Changes from previous version:
- Removed ElasticNet/Lasso meta-learner (was zeroing features)
- Single LSTM model making direct predictions
- Increased sequence length: 30 → 60 candles (15 hours of context)
- Kept 15m timeframe, 1 hour lookahead, 100 days data
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from data_scraper import DataScraper
from feature_engineering import FeatureEngineer
from models_lstm_simple import SimpleLSTMModel


class SimpleLSTMPipeline:
    """Simple LSTM training pipeline without meta-learner"""
    
    def __init__(self, symbol="BTCUSDT", interval="15m"):
        """
        Initialize pipeline
        
        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            interval: Candle interval (default: 15m)
        """
        self.symbol = symbol
        self.interval = interval
        
        # Initialize components
        self.scraper = DataScraper(symbol=symbol, interval=interval)
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = RobustScaler()
        
        # LSTM model (initialized after knowing sequence params)
        self.model = None
        self.prepared_data = {}
        
    def fetch_data(self, lookback_days="100 days ago UTC"):
        """Step 1: Fetch historical data"""
        print(f"\n{'='*60}")
        print(f"STEP 1: FETCHING DATA")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Lookback: {lookback_days}")
        
        df = self.scraper.fetch_historical_(lookback_days)
        print(f"\n✓ Fetched {len(df)} candles")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def engineer_features(self, df, threshold_percent=0.2, lookahead_periods=4):
        """Step 2: Engineer features and create target labels"""
        print(f"\n{'='*60}")
        print(f"STEP 2: FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        # Create features (context data is None for historical training)
        df_features = self.feature_engineer.generate_all_features(
            df,
            order_book_context=None,
            ticker_context=None,
            funding_rate=None,
            long_short_ratio=None
        )
        print(f"✓ Created {len(df_features.columns)} total features")
        
        # Create binary directional targets
        df_features = self.feature_engineer.create_target_labels(
            df_features,
            threshold_percent=threshold_percent,
            lookahead_periods=lookahead_periods
        )
        
        # Drop NaN rows
        initial_count = len(df_features)
        df_features = df_features.dropna()
        dropped = initial_count - len(df_features)
        
        print(f"✓ Created target labels (lookahead={lookahead_periods} periods)")
        print(f"  Dropped {dropped} rows with NaN")
        print(f"  Final dataset: {len(df_features)} samples")
        
        return df_features
    
    def select_features(self, df, n_features=20):
        """Step 3: Select most important features"""
        print(f"\n{'='*60}")
        print(f"STEP 3: FEATURE SELECTION")
        print(f"{'='*60}")
        
        # Get feature columns (exclude OHLCV and target)
        feature_cols = self.feature_engineer.get_feature_columns(df)
        print(f"Original features: {len(feature_cols)}")
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Use Random Forest for feature importance
        print("Calculating feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        selected_features = importances.head(n_features)['feature'].tolist()
        
        print(f"\n📊 Top {n_features} Features:")
        for feat, imp in zip(selected_features, importances.head(n_features)['importance']):
            print(f"  {feat:.<30} {imp:.6f}")
        
        print(f"\n✓ Selected {len(selected_features)} features")
        
        return selected_features
    
    def preprocess_data(self, df, selected_features, sequence_length=60, test_size=0.2):
        """Step 4: Preprocess data and create sequences"""
        print(f"\n{'='*60}")
        print(f"STEP 4: PREPROCESSING & SEQUENCE CREATION")
        print(f"{'='*60}")
        
        # Extract features and target
        X = df[selected_features].values
        y = df['target'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Temporal split (important for time series!)
        split_idx = int(len(X) * (1 - test_size))
        
        print(f"\n{'='*60}")
        print(f"TEMPORAL DATA SPLIT")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"Training:   {split_idx} samples ({(1-test_size)*100:.1f}%)")
        print(f"Test:       {len(X) - split_idx} samples ({test_size*100:.1f}%)")
        print(f"{'='*60}")
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Calculate class weights
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
        
        print(f"Class distribution: {dict(zip(unique, counts))}")
        print(f"Class weights: {class_weights}")
        
        # Scale features
        print(f"\nScaling features with RobustScaler...")
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        print(f"✓ Features scaled")
        
        # Create temporal sequences
        print(f"\nCreating temporal sequences (length={sequence_length})...")
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        print(f"✓ Created sequences")
        print(f"  Train: {X_train_seq.shape} (samples, timesteps, features)")
        print(f"  Test:  {X_test_seq.shape}")
        
        # Store prepared data
        self.prepared_data = {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'class_weights': class_weights,
            'selected_features': selected_features
        }
        
        print(f"{'='*60}\n")
        
        return self.prepared_data
    
    def _create_sequences(self, X, y, sequence_length):
        """Create temporal sequences from time series data"""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, sequence_length=60, n_features=20,
                   lstm_units=64, dropout=0.3, l2_reg=0.01,
                   learning_rate=0.001, epochs=50, batch_size=32,
                   early_stopping_patience=10):
        """Step 5: Train LSTM model (NO META-LEARNER)"""
        print(f"\n{'='*60}")
        print(f"STEP 5: TRAINING LSTM MODEL (NO META-LEARNER)")
        print(f"{'='*60}")
        
        # Initialize model
        self.model = SimpleLSTMModel(
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        # Build architecture
        self.model.build(
            lstm_units=lstm_units,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )
        
        # Train model
        history = self.model.train(
            X_train=self.prepared_data['X_train'],
            y_train=self.prepared_data['y_train'],
            X_val=self.prepared_data['X_test'],
            y_val=self.prepared_data['y_test'],
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            class_weights=self.prepared_data['class_weights']
        )
        
        return history
    
    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"STEP 6: EVALUATING ON {dataset_name.upper()} SET")
        print(f"{'='*60}")
        
        # Get predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Print metrics
        print(f"\n📊 {dataset_name.upper()} PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"  Samples:    {len(y)}")
        print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS METRICS:")
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        print(f"\n  Class 0 (Down ⬇️):")
        print(f"    Precision: {precision_per_class[0]:.4f}")
        print(f"    Recall:    {recall_per_class[0]:.4f}")
        print(f"    F1 Score:  {f1_per_class[0]:.4f}")
        
        print(f"\n  Class 1 (Up ⬆️):")
        print(f"    Precision: {precision_per_class[1]:.4f}")
        print(f"    Recall:    {recall_per_class[1]:.4f}")
        print(f"    F1 Score:  {f1_per_class[1]:.4f}")
        
        # Confusion matrix
        print(f"\n{'='*60}")
        print(f"CONFUSION MATRIX:")
        print(f"{'='*60}")
        print(f"\n              Predicted")
        print(f"           Down    Up")
        print(f"Actual ┌──────┬──────┐")
        print(f"Down   │ {cm[0][0]:>4} │ {cm[0][1]:>4} │")
        print(f"Up     │ {cm[1][0]:>4} │ {cm[1][1]:>4} │")
        print(f"       └──────┴──────┘")
        
        # Classification report
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REPORT:")
        print(f"{'='*60}\n")
        print(classification_report(y, y_pred, target_names=['Down ⬇️', 'Up ⬆️']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def save_models(self):
        """Step 7: Save all models and artifacts"""
        print(f"\n{'='*60}")
        print(f"STEP 7: SAVING MODEL")
        print(f"{'='*60}")
        
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"data/models/{self.symbol}_{self.interval}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LSTM model
        self.model.save(model_dir)
        
        # Save preprocessor
        import joblib
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✓ Preprocessor saved to {preprocessor_path}")
        
        # Save feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.prepared_data['selected_features'], feature_columns_path)
        print(f"✓ Feature columns saved to {feature_columns_path}")
        print(f"  Number of features: {len(self.prepared_data['selected_features'])}")
        
        # Save metadata
        import json
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'timestamp': timestamp,
            'model_type': 'SimpleLSTM',
            'sequence_length': self.model.sequence_length,
            'n_features': len(self.prepared_data['selected_features']),
            'features': self.prepared_data['selected_features']
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
        
        print(f"\n✓ All models saved to: {model_dir}")
        
        return model_dir
    
    def run_full_pipeline(self, lookback_days="100 days ago UTC",
                         threshold_percent=0.2, lookahead_periods=4,
                         n_features=20, sequence_length=60,
                         lstm_units=64, dropout=0.3, l2_reg=0.01,
                         learning_rate=0.001, epochs=50, batch_size=32,
                         early_stopping_patience=10, save_models=True):
        """Run the complete training pipeline"""
        
        print(f"\n{'='*60}")
        print(f"🚀 SIMPLE LSTM TRAINING PIPELINE - STARTING")
        print(f"{'='*60}")
        print(f"  Model Type: Simple LSTM (NO META-LEARNER)")
        print(f"  Symbol: {self.symbol}")
        print(f"  Interval: {self.interval}")
        print(f"  Sequence Length: {sequence_length} candles")
        print(f"  Lookahead: {lookahead_periods} periods")
        print(f"{'='*60}\n")
        
        # Run pipeline steps
        df = self.fetch_data(lookback_days)
        df_features = self.engineer_features(df, threshold_percent, lookahead_periods)
        selected_features = self.select_features(df_features, n_features)
        self.preprocess_data(df_features, selected_features, sequence_length)
        
        # Train LSTM
        history = self.train_lstm(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=lstm_units,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience
        )
        
        # Evaluate
        train_metrics = self.evaluate(
            self.prepared_data['X_train'],
            self.prepared_data['y_train'],
            dataset_name="Train"
        )
        
        test_metrics = self.evaluate(
            self.prepared_data['X_test'],
            self.prepared_data['y_test'],
            dataset_name="Test"
        )
        
        # Save models
        model_dir = None
        if save_models:
            model_dir = self.save_models()
        
        # Print summary
        print(f"\n{'#'*60}")
        print(f"#  PIPELINE COMPLETE!")
        print(f"#  Symbol: {self.symbol} | Interval: {self.interval}")
        print("#")
        print("#  TRAINING SET:")
        print(f"#    Samples:   {len(self.prepared_data['y_train'])}")
        print(f"#    Accuracy:  {train_metrics['accuracy']*100:.2f}%")
        print(f"#    F1 Score:  {train_metrics['f1']:.4f}")
        print("#")
        print("#  TEST SET:")
        print(f"#    Samples:   {len(self.prepared_data['y_test'])}")
        print(f"#    Accuracy:  {test_metrics['accuracy']*100:.2f}%")
        print(f"#    F1 Score:  {test_metrics['f1']:.4f}")
        print("#")
        if model_dir:
            print(f"#  Models saved to: {model_dir}")
        print("#"*60 + "\n")
        
        return {
            'metrics': {'train': train_metrics, 'test': test_metrics},
            'model_dir': model_dir,
            'pipeline': self
        }


def main():
    """Main training script"""
    # Configuration
    SYMBOL = "BTCUSDT"
    INTERVAL = "15m"  # 15m timeframe (less noise)
    LOOKBACK = "100 days ago UTC"  # 100 days of data
    THRESHOLD = 0.2
    LOOKAHEAD = 4  # 4 candles * 15min = 1 hour lookahead
    N_FEATURES = 20
    SEQUENCE_LENGTH = 60  # Increased from 30 to 60 (15 hours of context)
    
    # LSTM hyperparameters
    LSTM_UNITS = 64
    DROPOUT = 0.3
    L2_REG = 0.01
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    
    print("\n" + "="*60)
    print("SIMPLE LSTM - NO META-LEARNER CONFIGURATION:")
    print("  - Model: LSTM (2-layer) → Direct Binary Prediction")
    print("  - NO ElasticNet/Lasso meta-learner")
    print(f"  - Timeframe: {INTERVAL} candles")
    print(f"  - Sequence Length: {SEQUENCE_LENGTH} candles (15 hours)")
    print(f"  - Lookahead: {LOOKAHEAD} periods (1 hour)")
    print(f"  - Training Data: {LOOKBACK}")
    print("="*60 + "\n")
    
    # Create and run pipeline
    pipeline = SimpleLSTMPipeline(symbol=SYMBOL, interval=INTERVAL)
    results = pipeline.run_full_pipeline(
        lookback_days=LOOKBACK,
        threshold_percent=THRESHOLD,
        lookahead_periods=LOOKAHEAD,
        n_features=N_FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        lstm_units=LSTM_UNITS,
        dropout=DROPOUT,
        l2_reg=L2_REG,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_models=True
    )
    
    print("Training complete!")
    print(f"Model directory: {results['model_dir']}")


if __name__ == "__main__":
    main()
