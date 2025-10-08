"""
LSTM Training Pipeline - Temporal Sequence Modeling
Complete pipeline for training LSTM/GRU + Lasso meta-learner

Architecture:
  Base Model: LSTM/GRU (learns from 30-candle sequences)
  Meta-Learner: Lasso Regression (L1 regularization for feature selection)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_scraper import DataScraper
from feature_engineering import FeatureEngineer
from models_lstm import LSTMEnsemble


class LSTMPipeline:
    """Complete LSTM training pipeline with temporal sequences"""
    
    def __init__(self, symbol="BTCUSDT", interval="1m", use_gru=False):
        self.symbol = symbol
        self.interval = interval
        self.use_gru = use_gru  # GRU is faster, LSTM is slightly more accurate
        
        self.scraper = DataScraper()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = RobustScaler()
        
        # LSTM ensemble (initialized after knowing n_features)
        self.ensemble = None
        self.prepared_data = {}
        
    def fetch_data(self, lookback_days="35 days ago UTC"):
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
    
    def engineer_features(self, df, threshold_percent=0.2, lookahead_periods=15):
        """Step 2: Engineer features and create target labels"""
        print(f"\n{'='*60}")
        print(f"STEP 2: FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        # For historical training, we don't have live context data
        # These features will be None during training, but available during live prediction
        # The model will learn to work without them, and benefit from them when available
        
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
    
    def select_features(self, df_features, n_features=20):
        """Step 3: Feature selection using Random Forest importance"""
        print(f"\n{'='*60}")
        print(f"STEP 3: FEATURE SELECTION")
        print(f"{'='*60}")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(df_features)
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        print(f"Original features: {len(feature_cols)}")
        
        # Random Forest feature importance
        print("Calculating feature importance...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        print(f"\n📊 Top {n_features} Features:")
        for idx, row in feature_importance.head(n_features).iterrows():
            print(f"  {row['feature']:.<30} {row['importance']:.6f}")
        
        print(f"\n✓ Selected {len(top_features)} features")
        
        return df_features[top_features + ['target']], top_features
    
    def preprocess_data(self, df_features, selected_features, sequence_length=30):
        """
        Step 4: Preprocess data and create sequences
        
        Creates temporal sequences for LSTM/GRU:
          Each sample = [candle_t-30, candle_t-29, ..., candle_t]
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: PREPROCESSING & SEQUENCE CREATION")
        print(f"{'='*60}")
        
        # Prepare features and target
        X = df_features[selected_features].values
        y = df_features['target'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Check minimum samples
        min_samples = sequence_length * 10
        if len(X) < min_samples:
            raise ValueError(
                f"Insufficient data: {len(X)} samples. "
                f"Need at least {min_samples} for sequence_length={sequence_length}"
            )
        
        # Temporal split (80% train, 20% test)
        n_samples = len(X)
        train_size = int(n_samples * 0.8)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        print(f"\n{'='*60}")
        print(f"TEMPORAL DATA SPLIT")
        print(f"{'='*60}")
        print(f"Total samples: {n_samples}")
        print(f"Training:   {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"Test:       {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Calculate class weights
        unique_classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        class_weights = {
            int(cls): float(weight)
            for cls, weight in zip(unique_classes, class_weights_array)
        }
        
        from collections import Counter
        class_dist = Counter(y)
        print(f"Class distribution: {dict(class_dist)}")
        print(f"Class weights: {class_weights}\n")
        
        # Scale features
        print(f"Scaling features with RobustScaler...")
        self.preprocessor.fit(X_train)
        X_train_scaled = self.preprocessor.transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)
        print(f"✓ Features scaled\n")
        
        # Create sequences for LSTM/GRU
        print(f"Creating temporal sequences (length={sequence_length})...")
        X_train_seq, y_train_seq = self._create_sequences(
            X_train_scaled, y_train, sequence_length
        )
        X_test_seq, y_test_seq = self._create_sequences(
            X_test_scaled, y_test, sequence_length
        )
        
        print(f"✓ Created sequences")
        print(f"  Train: {X_train_seq.shape} (samples, timesteps, features)")
        print(f"  Test:  {X_test_seq.shape}")
        print(f"\n{'='*60}\n")
        
        # Store prepared data
        self.prepared_data = {
            'X_train': X_train_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'feature_columns': selected_features,
            'class_weights': class_weights,
            'sequence_length': sequence_length,
            'n_features': len(selected_features)
        }
        
        return self.prepared_data
    
    def _create_sequences(self, X, y, sequence_length):
        """
        Create temporal sequences for LSTM/GRU
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            sequence_length: Number of historical candles (e.g., 30)
        
        Returns:
            X_seq: Sequences (n_samples - sequence_length, sequence_length, n_features)
            y_seq: Targets (n_samples - sequence_length,)
        
        Example:
            If sequence_length=30, sample i contains candles [i-30:i]
            and predicts target at candle i
        """
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])  # Last 30 candles
            y_seq.append(y[i])  # Target at current candle
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm(self, lstm_units=64, dropout=0.3, l2_reg=0.01,
                   learning_rate=0.001, epochs=50, batch_size=32,
                   early_stopping_patience=10):
        """Step 5: Train LSTM/GRU base model"""
        print(f"\n{'='*60}")
        print(f"STEP 5: TRAINING {'GRU' if self.use_gru else 'LSTM'} BASE MODEL")
        print(f"{'='*60}")
        
        # Initialize ensemble
        self.ensemble = LSTMEnsemble(
            n_classes=2,
            sequence_length=self.prepared_data['sequence_length'],
            n_features=self.prepared_data['n_features'],
            use_gru=self.use_gru
        )
        
        # Build model
        self.ensemble.build(
            lstm_units=lstm_units,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )
        
        # Train model
        history = self.ensemble.train(
            self.prepared_data['X_train'],
            self.prepared_data['y_train'],
            X_val=None,  # Could add validation split here
            y_val=None,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience
        )
        
        return history
    
    def train_meta_learner(self, alpha=0.01, l1_ratio=0.5):
        """
        Step 6: Train ElasticNet meta-learner on LSTM outputs
        
        Args:
            alpha: Overall regularization strength (default 0.01)
            l1_ratio: Balance between L1 and L2 (default 0.5 = balanced)
        """
        print(f"\n{'='*60}")
        print(f"STEP 6: TRAINING ELASTICNET META-LEARNER")
        print(f"{'='*60}")
        
        self.ensemble.train_meta_learner(
            self.prepared_data['X_train'],
            self.prepared_data['y_train'],
            alpha=alpha,
            l1_ratio=l1_ratio
        )
        
        return self
    
    def evaluate(self, dataset='test'):
        """Step 7: Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"STEP 7: EVALUATING ON {dataset.upper()} SET")
        print(f"{'='*60}")
        
        if dataset == 'test':
            X = self.prepared_data['X_test']
            y_true = self.prepared_data['y_test']
        else:
            X = self.prepared_data['X_train']
            y_true = self.prepared_data['y_train']
        
        # Get predictions
        y_pred = self.ensemble.predict(X)
        y_proba = self.ensemble.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\n📊 {dataset.upper()} PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"  Samples:    {len(y_true)}")
        print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS METRICS:")
        class_names = ['Down ⬇️', 'Up ⬆️']
        for i, class_name in enumerate(class_names):
            class_precision = precision_score(
                y_true == i, y_pred == i, zero_division=0
            )
            class_recall = recall_score(
                y_true == i, y_pred == i, zero_division=0
            )
            class_f1 = f1_score(
                y_true == i, y_pred == i, zero_division=0
            )
            
            print(f"\n  Class {i} ({class_name}):")
            print(f"    Precision: {class_precision:.4f}")
            print(f"    Recall:    {class_recall:.4f}")
            print(f"    F1 Score:  {class_f1:.4f}")
        
        # Confusion matrix
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX:")
        print(f"{'='*60}\n")
        
        cm = confusion_matrix(y_true, y_pred)
        
        print("              Predicted")
        print("           Down    Up")
        print("Actual ┌──────┬──────┐")
        print(f"Down   │ {cm[0,0]:>4} │ {cm[0,1]:>4} │")
        print(f"Up     │ {cm[1,0]:>4} │ {cm[1,1]:>4} │")
        print("       └──────┴──────┘")
        
        # Classification report
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT:")
        print(f"{'='*60}\n")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def save_model(self, save_dir='data/models', threshold_percent=0.2,
                   lookahead_periods=15, train_metrics=None, test_metrics=None):
        """Step 8: Save trained model and metadata"""
        print(f"\n{'='*60}")
        print(f"STEP 8: SAVING MODEL")
        print(f"{'='*60}")
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(
            save_dir,
            f"{self.symbol}_{self.interval}_{timestamp}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LSTM ensemble (base model + meta-learner)
        self.ensemble.save(model_dir)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✓ Preprocessor saved to {preprocessor_path}")
        
        # Save feature columns
        feature_path = os.path.join(model_dir, "feature_columns.pkl")
        joblib.dump(self.prepared_data['feature_columns'], feature_path)
        print(f"✓ Feature columns saved to {feature_path}")
        print(f"  Number of features: {len(self.prepared_data['feature_columns'])}")
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'model_type': 'GRU' if self.use_gru else 'LSTM',
            'n_features': self.prepared_data['n_features'],
            'n_classes': 2,
            'sequence_length': self.prepared_data['sequence_length'],
            'training_date': timestamp,
            'class_weights': self.prepared_data['class_weights'],
            'model_architecture': f"{'GRU' if self.use_gru else 'LSTM'} + Lasso Meta-Learner (Temporal Sequences)",
            'threshold_percent': threshold_percent,
            'lookahead_periods': lookahead_periods,
            'use_stacking': self.ensemble.use_stacking,
            'has_meta_learner': self.ensemble.meta_learner is not None
        }
        
        if train_metrics:
            metadata['performance'] = {
                'train': train_metrics,
                'test': test_metrics if test_metrics else {}
            }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
        
        print(f"\n✓ All models saved to: {model_dir}")
        
        return model_dir
    
    def run_full_pipeline(self, lookback_days="35 days ago UTC",
                          threshold_percent=0.2, lookahead_periods=15,
                          n_features=20, sequence_length=30,
                          lstm_units=64, dropout=0.3, l2_reg=0.01,
                          learning_rate=0.001, epochs=50, batch_size=32,
                          early_stopping_patience=10, 
                          elasticnet_alpha=0.001, elasticnet_l1_ratio=0.5,
                          save_models=True):
        """Run complete LSTM training pipeline"""
        
        print("\n" + "="*60)
        print("🚀 LSTM TRAINING PIPELINE - STARTING")
        print("="*60)
        print(f"  Model Type: {'GRU' if self.use_gru else 'LSTM'} + ElasticNet Meta-Learner")
        print(f"  Symbol: {self.symbol}")
        print(f"  Interval: {self.interval}")
        print(f"  Sequence Length: {sequence_length} candles")
        print(f"  Lookahead: {lookahead_periods} periods")
        print("="*60 + "\n")
        
        # Step 1: Fetch data
        df = self.fetch_data(lookback_days)
        
        # Step 2: Engineer features
        df_features = self.engineer_features(df, threshold_percent, lookahead_periods)
        
        # Step 3: Feature selection
        df_selected, selected_features = self.select_features(df_features, n_features)
        
        # Step 4: Preprocess and create sequences
        self.preprocess_data(df_selected, selected_features, sequence_length)
        
        # Step 5: Train LSTM/GRU
        history = self.train_lstm(
            lstm_units=lstm_units,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience
        )
        
        # Step 6: Train meta-learner
        self.train_meta_learner(
            alpha=elasticnet_alpha,
            l1_ratio=elasticnet_l1_ratio
        )
        
        # Step 7: Evaluate
        train_metrics = self.evaluate('train')
        test_metrics = self.evaluate('test')
        
        # Step 8: Save model
        model_dir = None
        if save_models:
            model_dir = self.save_model(
                threshold_percent=threshold_percent,
                lookahead_periods=lookahead_periods,
                train_metrics=train_metrics,
                test_metrics=test_metrics
            )
        
        print("\n" + "#"*60)
        print("#  PIPELINE COMPLETE!")
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
    INTERVAL = "15m"  # Changed from 1m to 15m (less noise)
    LOOKBACK = "100 days ago UTC"  # Changed from 35 to 100 days (more data)
    THRESHOLD = 0.2
    LOOKAHEAD = 4  # 4 candles * 15min = 1 hour lookahead (changed from 15 min)
    N_FEATURES = 20
    SEQUENCE_LENGTH = 30
    USE_GRU = False  # Set True for GRU (faster), False for LSTM (more accurate)
    
    # LSTM hyperparameters
    LSTM_UNITS = 64
    DROPOUT = 0.3
    L2_REG = 0.01
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    
    # Meta-learner hyperparameters
    ELASTICNET_ALPHA = 0.001  # Overall regularization (reduced from 0.01)
    ELASTICNET_L1_RATIO = 0.5  # 50% L1 (selection), 50% L2 (stability)
    
    print("\n" + "="*60)
    print("LSTM + ELASTICNET META-LEARNER - CONFIGURATION:")
    print("  - Base Model: " + ("GRU" if USE_GRU else "LSTM"))
    print("  - Meta-Learner: ElasticNet (L1 + L2 regularization)")
    print("  - Binary Classification: 0=Down, 1=Up")
    print(f"  - Timeframe: {INTERVAL} candles")
    print(f"  - Sequence Length: {SEQUENCE_LENGTH} candles")
    print(f"  - Lookahead: {LOOKAHEAD} periods ({LOOKAHEAD * 15} minutes = 1 hour)")
    print(f"  - Training Data: {LOOKBACK}")
    print(f"  - ElasticNet Alpha: {ELASTICNET_ALPHA} (strength)")
    print(f"  - ElasticNet L1 Ratio: {ELASTICNET_L1_RATIO} ({ELASTICNET_L1_RATIO*100:.0f}% L1, {(1-ELASTICNET_L1_RATIO)*100:.0f}% L2)")
    print("="*60 + "\n")
    
    # Create and run pipeline
    pipeline = LSTMPipeline(symbol=SYMBOL, interval=INTERVAL, use_gru=USE_GRU)
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
        elasticnet_alpha=ELASTICNET_ALPHA,
        elasticnet_l1_ratio=ELASTICNET_L1_RATIO,
        save_models=True
    )
    
    print("Training complete!")
    print(f"Model directory: {results['model_dir']}")


if __name__ == "__main__":
    main()
