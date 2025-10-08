"""
ML Training Pipeline
Complete pipeline for training the ensemble model
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_scraper import DataScraper
from feature_engineering import FeatureEngineer
from preprocessing import DataPreprocessor
from models import CatBoostModel, RandomForestModel, LogisticRegressionModel, EnsembleModel


class MLPipeline:
    """Complete ML training pipeline"""
    
    def __init__(self, symbol="BTCUSDT", interval="30s"):
        self.symbol = symbol
        self.interval = interval
        self.scraper = None
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.ensemble = EnsembleModel(n_classes=3)
        self.prepared_data = None
        
    def fetch_data(self, lookback_days="2 days ago UTC"):
        """Step 1: Fetch data from Binance"""
        print(f"\n{'='*60}")
        print(f"STEP 1: DATA COLLECTION")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Lookback: {lookback_days}")
        
        self.scraper = DataScraper(symbol=self.symbol, interval=self.interval)
        df = self.scraper.fetch_historical_(lookback_days)
        
        print(f"✓ Fetched {len(df)} candlesticks")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")
        print(f"  Shape: {df.shape}")
        
        return df
    
    def engineer_features(self, df, threshold_percent=0.5, lookahead_periods=6):
        """Step 2: Feature engineering"""
        print(f"\n{'='*60}")
        print(f"STEP 2: FEATURE ENGINEERING")
        print(f"{'='*60}")
        print(f"Movement threshold: {threshold_percent}%")
        print(f"Lookahead periods: {lookahead_periods}")
        
        # Get market context if available
        order_book_context = None
        ticker_context = None
        if self.scraper:
            try:
                _, context = self.scraper.fetch_context_data()
                order_book_context = context.get('order_book')
                ticker_context = context.get('ticker')
            except Exception as e:
                print(f"  Warning: Could not fetch market context: {e}")
        
        # Generate all features
        df_features = self.feature_engineer.generate_all_features(
            df, 
            order_book_context=order_book_context,
            ticker_context=ticker_context,
            create_targets=True,
            threshold_percent=threshold_percent,
            lookahead_periods=lookahead_periods
        )
        
        return df_features
    
    def select_important_features(self, df_features, n_features=20):
        """Step 2.5: Feature Selection - Keep only the most important features"""
        print(f"\n{'='*60}")
        print(f"STEP 2.5: FEATURE SELECTION")
        print(f"{'='*60}")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        # Get feature columns and target
        feature_cols = self.feature_engineer.get_feature_columns(df_features)
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        print(f"Original features: {len(feature_cols)}")
        
        # Method 1: Random Forest Feature Importance
        print("\nCalculating feature importance...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Get top N features
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        print(f"\n📊 Top {n_features} Most Important Features:")
        for idx, row in feature_importance.head(n_features).iterrows():
            print(f"  {row['feature']:40s}: {row['importance']:.6f}")
        
        print(f"\n✓ Selected {len(top_features)} features (reduced from {len(feature_cols)})")
        print(f"  Total importance captured: {feature_importance.head(n_features)['importance'].sum():.4f}")
        
        return top_features
    
    def preprocess_data(self, df_features, undersample=True, sequence_length=30, selected_features=None):
        """Step 3: Data preprocessing"""
        print(f"\n{'='*60}")
        print(f"STEP 3: DATA PREPROCESSING")
        print(f"{'='*60}")
        
        # Check minimum dataset size
        if len(df_features) < 1000:
            raise ValueError(
                f"Dataset too small! Got {len(df_features)} samples, need at least 1000 samples.\n"
                f"Please collect more data (e.g., increase lookback period)."
            )
        
        # Get feature columns (use selected features if provided)
        if selected_features:
            feature_cols = selected_features
            print(f"Using {len(feature_cols)} selected features")
        else:
            feature_cols = self.feature_engineer.get_feature_columns(df_features)
            print(f"Using all {len(feature_cols)} features")
        
        # Prepare data with 80/20 split - DISABLE undersampling to preserve data
        self.prepared_data = self.preprocessor.prepare_data_pipeline(
            df_features,
            feature_cols,
            target_col='target',
            train_ratio=0.8,  # 80% training, 20% testing
            val_ratio=0.0,  # No validation set
            undersample=False,  # DISABLED - was too aggressive, causing severe data loss
            undersample_ratio=0.7,  # Not used when undersample=False
            scaler_type='robust',
            sequence_length=sequence_length
        )
        
        return self.prepared_data
    
    def train_catboost(self):
        """Step 4a: Train CatBoost"""
        print(f"\n{'='*60}")
        print(f"STEP 4a: TRAINING CATBOOST")
        print(f"{'='*60}")
        
        self.ensemble.catboost.build(
            class_weights=self.prepared_data['class_weights'],
            iterations=500,  # Reduced from 1000
            depth=4,  # Reduced from 6
            learning_rate=0.01,  # Reduced from 0.03
            l2_leaf_reg=10,  # L2 regularization
            bagging_temperature=1.0,  # Regularization via bagging
            random_strength=2.0  # Increased randomness
        )
        
        # Check if we have validation set
        has_val = len(self.prepared_data.get('X_val', [])) > 0
        
        if has_val:
            self.ensemble.catboost.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                self.prepared_data['X_val'],
                self.prepared_data['y_val']
            )
            # Evaluate on validation set
            y_pred = self.ensemble.catboost.predict(self.prepared_data['X_val'])
            f1 = f1_score(self.prepared_data['y_val'], y_pred, average='macro')
            print(f"\n✓ CatBoost Validation F1 Score: {f1:.4f}")
        else:
            # Train without validation set
            self.ensemble.catboost.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                None,  # No validation
                None
            )
            # Evaluate on test set instead
            y_pred = self.ensemble.catboost.predict(self.prepared_data['X_test'])
            f1 = f1_score(self.prepared_data['y_test'], y_pred, average='macro')
            print(f"\n✓ CatBoost Test F1 Score: {f1:.4f}")
        
    def train_random_forest(self):
        """Step 4b: Train Random Forest"""
        print(f"\n{'='*60}")
        print(f"STEP 4b: TRAINING RANDOM FOREST")
        print(f"{'='*60}")
        
        self.ensemble.random_forest.build(
            class_weights=self.prepared_data['class_weights'],
            n_estimators=100,  # Reduced from 200
            max_depth=8,  # Reduced from 15
            min_samples_split=20,  # Increased from default
            min_samples_leaf=10,  # Increased from default
            max_features='sqrt'  # Limit features per tree
        )
        
        # Check if we have validation set
        has_val = len(self.prepared_data.get('X_val', [])) > 0
        
        if has_val:
            self.ensemble.random_forest.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                self.prepared_data['X_val'],
                self.prepared_data['y_val']
            )
            # Evaluate on validation set
            y_pred = self.ensemble.random_forest.predict(self.prepared_data['X_val'])
            f1 = f1_score(self.prepared_data['y_val'], y_pred, average='macro')
            print(f"\n✓ Random Forest Validation F1 Score: {f1:.4f}")
        else:
            # Train without validation set
            self.ensemble.random_forest.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                None,  # No validation
                None
            )
            # Evaluate on test set instead
            y_pred = self.ensemble.random_forest.predict(self.prepared_data['X_test'])
            f1 = f1_score(self.prepared_data['y_test'], y_pred, average='macro')
            print(f"\n✓ Random Forest Test F1 Score: {f1:.4f}")
    
    def train_logistic(self):
        """Step 4c: Train Logistic Regression"""
        print(f"\n{'='*60}")
        print(f"STEP 4c: TRAINING LOGISTIC REGRESSION")
        print(f"{'='*60}")
        
        self.ensemble.logistic.build(
            class_weights=True,
            C=1.0,  # Regularization strength
            max_iter=1000,
            solver='lbfgs'
        )
        
        # Check if we have validation set
        has_val = len(self.prepared_data.get('X_val', [])) > 0
        
        if has_val:
            self.ensemble.logistic.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                self.prepared_data['X_val'],
                self.prepared_data['y_val']
            )
            # Evaluate on validation set
            y_pred = self.ensemble.logistic.predict(self.prepared_data['X_val'])
            f1 = f1_score(self.prepared_data['y_val'], y_pred, average='macro')
            print(f"\n✓ Logistic Regression Validation F1 Score: {f1:.4f}")
        else:
            # Train without validation set
            self.ensemble.logistic.train(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                None,  # No validation
                None
            )
            # Evaluate on test set instead
            y_pred = self.ensemble.logistic.predict(self.prepared_data['X_test'])
            f1 = f1_score(self.prepared_data['y_test'], y_pred, average='macro')
            print(f"\n✓ Logistic Regression Test F1 Score: {f1:.4f}")
    
    def train_meta_learner(self):
        """Step 4d: Train Meta-Learner (Stacking)"""
        print(f"\n{'='*60}")
        print(f"STEP 4d: TRAINING META-LEARNER (STACKING)")
        print(f"{'='*60}")
        
        # Check if we have validation set
        has_val = len(self.prepared_data.get('X_val', [])) > 0
        
        if has_val:
            # Train meta-learner with validation set
            self.ensemble.train_meta_learner(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                self.prepared_data['X_val'],
                self.prepared_data['y_val']
            )
        else:
            # Train meta-learner without validation
            self.ensemble.train_meta_learner(
                self.prepared_data['X_train'],
                self.prepared_data['y_train'],
                None,
                None
            )
        
    def train_lstm(self):
        """Step 4d: Train LSTM"""
        print(f"\n{'='*60}")
        print(f"STEP 4d: TRAINING LSTM")
        print(f"{'='*60}")
        
        # Get input shape
        input_shape = (
            self.prepared_data['X_train_lstm'].shape[1],  # timesteps
            self.prepared_data['X_train_lstm'].shape[2]   # features
        )
        
        self.ensemble.lstm.build(
            input_shape=input_shape,
            lstm_units=[64, 32],  # Reduced from [128, 64]
            dropout=0.5,  # Increased from 0.3
            recurrent_dropout=0.3,  # Add recurrent dropout
            l2_reg=0.01  # L2 regularization
        )
        
        self.ensemble.lstm.compile(
            learning_rate=0.001,
            class_weights=self.prepared_data['class_weights']
        )
        
        # Check if we have validation set
        has_val = len(self.prepared_data.get('X_val_lstm', [])) > 0
        
        if has_val:
            self.ensemble.lstm.train(
                self.prepared_data['X_train_lstm'],
                self.prepared_data['y_train_lstm'],
                self.prepared_data['X_val_lstm'],
                self.prepared_data['y_val_lstm'],
                class_weights=self.prepared_data['class_weights'],
                epochs=50,
                batch_size=32
            )
            # Evaluate on validation set
            y_pred = self.ensemble.lstm.predict(self.prepared_data['X_val_lstm'])
            f1 = f1_score(self.prepared_data['y_val_lstm'], y_pred, average='macro')
            print(f"\n✓ LSTM Validation F1 Score: {f1:.4f}")
        else:
            # Train with test set as validation for early stopping
            self.ensemble.lstm.train(
                self.prepared_data['X_train_lstm'],
                self.prepared_data['y_train_lstm'],
                self.prepared_data['X_test_lstm'],
                self.prepared_data['y_test_lstm'],
                class_weights=self.prepared_data['class_weights'],
                epochs=50,
                batch_size=32
            )
            # Evaluate on test set
            y_pred = self.ensemble.lstm.predict(self.prepared_data['X_test_lstm'])
            f1 = f1_score(self.prepared_data['y_test_lstm'], y_pred, average='macro')
            print(f"\n✓ LSTM Test F1 Score: {f1:.4f}")
        
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, dataset_name="Dataset"):
        """Calculate comprehensive metrics for a dataset"""
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (multi-class)
        try:
            roc_auc_ovr = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            roc_auc_ovo = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
        except Exception as e:
            print(f"  Warning: Could not calculate ROC AUC for {dataset_name}: {e}")
            roc_auc_ovr = 0.0
            roc_auc_ovo = 0.0
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'roc_auc_ovr': roc_auc_ovr,
            'roc_auc_ovo': roc_auc_ovo,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist()
            }
        }
    
    def _print_metrics(self, metrics, dataset_name="Dataset", y_true=None, y_pred=None, dataset_size=None):
        """Print metrics in a formatted way"""
        
        print(f"\n{'='*60}")
        print(f"{dataset_name.upper()} PERFORMANCE METRICS")
        print(f"{'='*60}")
        if dataset_size is not None:
            print(f"\n📦 DATASET SIZE: {dataset_size} samples")
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision (Macro):     {metrics['precision_macro']:.4f}")
        print(f"  Precision (Weighted):  {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro):        {metrics['recall_macro']:.4f}")
        print(f"  Recall (Weighted):     {metrics['recall_weighted']:.4f}")
        print(f"  F1 Score (Macro):      {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (Weighted):   {metrics['f1_weighted']:.4f}")
        print(f"  ROC AUC (OvR):         {metrics['roc_auc_ovr']:.4f}")
        print(f"  ROC AUC (OvO):         {metrics['roc_auc_ovo']:.4f}")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS METRICS:")
        class_names = ['No Movement', 'Large Up', 'Large Down']
        precision_list = metrics['per_class_metrics']['precision']
        recall_list = metrics['per_class_metrics']['recall']
        f1_list = metrics['per_class_metrics']['f1']
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_list):
                print(f"\n  Class {i} ({class_name}):")
                print(f"    Precision: {precision_list[i]:.4f}")
                print(f"    Recall:    {recall_list[i]:.4f}")
                print(f"    F1 Score:  {f1_list[i]:.4f}")
            else:
                print(f"\n  Class {i} ({class_name}): No samples in dataset")
        
        # Classification report
        if y_true is not None and y_pred is not None:
            print(f"\n{'='*60}")
            print("DETAILED CLASSIFICATION REPORT:")
            print(f"{'='*60}")
            # Get unique classes in y_true and y_pred
            unique_classes = sorted(set(list(y_true) + list(y_pred)))
            all_class_names = ['No Movement', 'Large Up', 'Large Down']
            target_names_subset = [all_class_names[i] for i in unique_classes]
            
            print(classification_report(
                y_true, 
                y_pred,
                labels=unique_classes,
                target_names=target_names_subset,
                zero_division=0
            ))
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX:")
        print(f"{'='*60}")
        
        # Determine number of classes in confusion matrix
        n_classes_cm = cm.shape[0]
        all_class_names = ['No Mov', 'Up    ', 'Down  ']
        class_names_subset = all_class_names[:n_classes_cm]
        
        # Print header
        print("\n              Predicted")
        header = "           "
        for i in range(n_classes_cm):
            if i == 0:
                header += "No Mov"
            elif i == 1:
                header += "    Up"
            elif i == 2:
                header += "  Down"
        print(header)
        
        # Print separator
        sep = "Actual ┌─────┬──────"
        for i in range(n_classes_cm - 1):
            sep += "┬────"
        sep += "─┐"
        print(sep)
        
        # Print rows
        for i in range(n_classes_cm):
            row_str = f"{class_names_subset[i]} │     │"
            for j in range(n_classes_cm):
                row_str += f" {cm[i][j]:4d} │"
            print(row_str)
        
        # Print bottom
        bottom = "       └─────┴──────"
        for i in range(n_classes_cm - 1):
            bottom += "┴────"
        bottom += "─┘"
        print(bottom)
    
    def evaluate_ensemble(self):
        """Step 5: Evaluate ensemble on train and test sets (3-model ensemble without LSTM)"""
        print(f"\n{'='*60}")
        print(f"STEP 5: ENSEMBLE EVALUATION")
        print(f"{'='*60}")
        
        # Set ensemble weights for 3-model ensemble
        self.ensemble.set_weights(catboost_weight=0.5, rf_weight=0.25, logistic_weight=0.25)
        
        # ===== TRAINING SET EVALUATION =====
        print(f"\n{'#'*60}")
        print(f"# EVALUATING ON TRAINING SET")
        print(f"{'#'*60}")
        
        y_pred_train = self.ensemble.predict(self.prepared_data['X_train'])
        y_pred_proba_train = self.ensemble.predict_proba(self.prepared_data['X_train'])
        
        train_metrics = self._calculate_metrics(
            self.prepared_data['y_train'], y_pred_train, y_pred_proba_train, "Training"
        )
        self._print_metrics(train_metrics, "Training", self.prepared_data['y_train'], y_pred_train, dataset_size=len(self.prepared_data['y_train']))
        
        # Check if we have validation set or just test set
        has_validation = self.prepared_data.get('X_val') is not None and len(self.prepared_data.get('X_val', [])) > 0
        
        if has_validation:
            # ===== VALIDATION SET EVALUATION =====
            print(f"\n{'#'*60}")
            print(f"# EVALUATING ON VALIDATION SET")
            print(f"{'#'*60}")
            
            y_pred_val = self.ensemble.predict(self.prepared_data['X_val'])
            y_pred_proba_val = self.ensemble.predict_proba(self.prepared_data['X_val'])
            
            val_metrics = self._calculate_metrics(
                self.prepared_data['y_val'], y_pred_val, y_pred_proba_val, "Validation"
            )
            self._print_metrics(val_metrics, "Validation", self.prepared_data['y_val'], y_pred_val, dataset_size=len(self.prepared_data['y_val']))
        else:
            val_metrics = None
        
        # ===== TEST SET EVALUATION =====
        print(f"\n{'#'*60}")
        print(f"# EVALUATING ON TEST SET")
        print(f"{'#'*60}")
        
        y_pred_test = self.ensemble.predict(self.prepared_data['X_test'])
        y_pred_proba_test = self.ensemble.predict_proba(self.prepared_data['X_test'])
        
        test_metrics = self._calculate_metrics(
            self.prepared_data['y_test'], y_pred_test, y_pred_proba_test, "Test"
        )
        self._print_metrics(test_metrics, "Test", self.prepared_data['y_test'], y_pred_test, dataset_size=len(self.prepared_data['y_test']))
        
        # ===== OVERFITTING ANALYSIS =====
        print(f"\n{'='*60}")
        print(f"🔍 OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        
        # Compare train vs test (or validation if available)
        comparison_set = val_metrics if has_validation else test_metrics
        comparison_name = "Validation" if has_validation else "Test"
        
        train_comp_acc_gap = train_metrics['accuracy'] - comparison_set['accuracy']
        train_comp_f1_gap = train_metrics['f1_macro'] - comparison_set['f1_macro']
        train_comp_roc_gap = train_metrics['roc_auc_ovr'] - comparison_set['roc_auc_ovr']
        
        print(f"\n📊 Train vs {comparison_name} Gap:")
        print(f"  Accuracy Gap:  {train_comp_acc_gap:+.4f} ({train_comp_acc_gap*100:+.2f}%)")
        print(f"  F1 Score Gap:  {train_comp_f1_gap:+.4f}")
        print(f"  ROC AUC Gap:   {train_comp_roc_gap:+.4f}")
        
        # Overfitting verdict
        print(f"\n🎯 Overfitting Assessment:")
        if abs(train_comp_acc_gap) < 0.05:
            print(f"  ✅ GOOD: Minimal overfitting (accuracy gap < 5%)")
        elif abs(train_comp_acc_gap) < 0.10:
            print(f"  ⚠️  MODERATE: Some overfitting (accuracy gap 5-10%)")
        else:
            print(f"  🔴 SEVERE: Significant overfitting (accuracy gap > 10%)")
        
        if abs(train_comp_f1_gap) < 0.05:
            print(f"  ✅ GOOD: F1 score gap < 0.05")
        elif abs(train_comp_f1_gap) < 0.10:
            print(f"  ⚠️  MODERATE: F1 score gap 0.05-0.10")
        else:
            print(f"  🔴 SEVERE: F1 score gap > 0.10")
        
        print(f"\n{'='*60}")
        print(f"✓ Ensemble Evaluation Complete")
        print(f"  - Training Samples:   {len(self.prepared_data['y_train'])}")
        if has_validation:
            print(f"  - Validation Samples: {len(self.prepared_data['y_val'])}")
        print(f"  - Test Samples:       {len(self.prepared_data['y_test'])}")
        print(f"  - Training Accuracy:  {train_metrics['accuracy']*100:.2f}%")
        if has_validation:
            print(f"  - Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
        print(f"  - Test Accuracy:      {test_metrics['accuracy']*100:.2f}%")
        print(f"{'='*60}\n")
        
        result = {
            'train': train_metrics,
            'test': test_metrics,
            'overfitting_analysis': {
                'train_test_accuracy_gap': float(train_comp_acc_gap),
                'train_test_f1_gap': float(train_comp_f1_gap),
                'train_test_roc_gap': float(train_comp_roc_gap),
            }
        }
        
        if has_validation:
            result['validation'] = val_metrics
        
        return result
        
    def save_models(self, save_dir='data/models', metrics=None, threshold_percent=0.2, lookahead_periods=6):
        """Step 6: Save models"""
        print(f"\n{'='*60}")
        print(f"STEP 6: SAVING MODELS")
        print(f"{'='*60}")
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(save_dir, f"{self.symbol}_{self.interval}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble models
        self.ensemble.save(model_dir)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✓ Preprocessor saved to {preprocessor_path}")
        
        # Save feature columns
        feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.prepared_data['feature_columns'], feature_cols_path)
        print(f"✓ Feature columns saved to {feature_cols_path}")
        print(f"  Number of features: {len(self.prepared_data['feature_columns'])}")
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'n_features': len(self.prepared_data['feature_columns']),
            'n_classes': 3,
            'training_date': timestamp,
            'class_weights': self.prepared_data['class_weights'],
            'model_architecture': 'Stacked Ensemble: (CatBoost + Random Forest + Logistic) → Meta-Learner',
            'ensemble_weights': self.ensemble.weights if self.ensemble.weights else {'catboost': 0.5, 'rf': 0.25, 'logistic': 0.25},
            'use_stacking': self.ensemble.use_stacking,
            'has_meta_learner': self.ensemble.meta_learner is not None,
            'threshold_percent': threshold_percent,
            'lookahead_periods': lookahead_periods,
            # Enhanced metadata for contextual analysis
            'contextual_thresholds': {
                'high_volatility_atr_pct': 1.5,
                'very_high_volatility_atr_pct': 2.5,
                'overbought_rsi': 70,
                'oversold_rsi': 30,
                'strong_momentum_roc': 2.0,
                'high_volume_ratio': 1.5,
                'tight_bollinger': 0.015,
                'wide_bollinger': 0.035,
                'strong_trend_adx': 40,
                'weak_trend_adx': 20,
                'short_trend_threshold': 0.1,
                'long_trend_threshold': 0.15
            },
            'trend_config': {
                'short_term_smas': [7, 14],
                'long_term_smas': [21, 50],
                'adx_periods': 14,
                'macd_config': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            'action_thresholds': {
                'high_confidence': 0.7,
                'medium_confidence': 0.5,
                'min_trend_score': 2,
                'high_conviction_score': 4
            }
        }
        
        # Add performance metrics if provided
        if metrics:
            metadata['performance'] = {
                'train': {
                    'accuracy': float(metrics['train']['accuracy']),
                    'precision_macro': float(metrics['train']['precision_macro']),
                    'recall_macro': float(metrics['train']['recall_macro']),
                    'f1_macro': float(metrics['train']['f1_macro']),
                    'roc_auc_ovr': float(metrics['train']['roc_auc_ovr'])
                },
                'test': {
                    'accuracy': float(metrics['test']['accuracy']),
                    'precision_macro': float(metrics['test']['precision_macro']),
                    'recall_macro': float(metrics['test']['recall_macro']),
                    'f1_macro': float(metrics['test']['f1_macro']),
                    'roc_auc_ovr': float(metrics['test']['roc_auc_ovr'])
                },
                'overfitting_analysis': metrics['overfitting_analysis']
            }
            
            # Add cross-validation results if available
            if 'cv_results' in metrics:
                metadata['performance']['cross_validation'] = {
                    'accuracy_mean': float(metrics['cv_results']['accuracy']['mean']),
                    'accuracy_std': float(metrics['cv_results']['accuracy']['std']),
                    'f1_mean': float(metrics['cv_results']['f1_macro']['mean']),
                    'f1_std': float(metrics['cv_results']['f1_macro']['std']),
                    'precision_mean': float(metrics['cv_results']['precision_macro']['mean']),
                    'recall_mean': float(metrics['cv_results']['recall_macro']['mean'])
                }
            
            # Add validation metrics if available
            if 'validation' in metrics:
                metadata['performance']['validation'] = {
                    'accuracy': float(metrics['validation']['accuracy']),
                    'precision_macro': float(metrics['validation']['precision_macro']),
                    'recall_macro': float(metrics['validation']['recall_macro']),
                    'f1_macro': float(metrics['validation']['f1_macro']),
                    'roc_auc_ovr': float(metrics['validation']['roc_auc_ovr'])
                }
        
        import json
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
        
        print(f"\n✓ All models saved to: {model_dir}")
        return model_dir
    
    def run_kfold_validation(self, n_splits=5):
        """Perform k-fold cross-validation to assess model generalization"""
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
        print(f"{'='*60}")
        
        # Prepare data for k-fold
        X = self.prepared_data['X_train']
        y = self.prepared_data['y_train']
        
        # Initialize k-fold splitter (stratified to preserve class distribution)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = {
            'accuracy': [],
            'f1_macro': [],
            'precision_macro': [],
            'recall_macro': []
        }
        
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}:")
            
            # Split data
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Train a simple Random Forest for quick validation
            from sklearn.ensemble import RandomForestClassifier
            fold_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            fold_model.fit(X_fold_train, y_fold_train)
            y_pred = fold_model.predict(X_fold_val)
            
            # Calculate metrics
            acc = accuracy_score(y_fold_val, y_pred)
            f1 = f1_score(y_fold_val, y_pred, average='macro', zero_division=0)
            prec = precision_score(y_fold_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_fold_val, y_pred, average='macro', zero_division=0)
            
            fold_scores['accuracy'].append(acc)
            fold_scores['f1_macro'].append(f1)
            fold_scores['precision_macro'].append(prec)
            fold_scores['recall_macro'].append(rec)
            
            print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        # Calculate statistics
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        cv_results = {}
        for metric, scores in fold_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv_results[metric] = {
                'mean': mean_score,
                'std': std_score,
                'scores': scores
            }
            print(f"{metric.upper():20s}: {mean_score:.4f} (±{std_score:.4f})")
        
        # Variance check
        print(f"\n🎯 Generalization Assessment:")
        if cv_results['accuracy']['std'] < 0.05:
            print(f"  ✅ GOOD: Low variance ({cv_results['accuracy']['std']:.4f}) - Model generalizes well")
        elif cv_results['accuracy']['std'] < 0.10:
            print(f"  ⚠️  MODERATE: Medium variance ({cv_results['accuracy']['std']:.4f})")
        else:
            print(f"  🔴 HIGH: High variance ({cv_results['accuracy']['std']:.4f}) - Unstable model")
        
        return cv_results
    
    def run_full_pipeline(self, lookback_days="2 days ago UTC", 
                         threshold_percent=0.5, lookahead_periods=6,
                         save_models=True, run_cv=True):
        """Run the complete pipeline"""
        print(f"\n{'#'*60}")
        print(f"#  ML TRAINING PIPELINE")
        print(f"#  Symbol: {self.symbol} | Interval: {self.interval}")
        print(f"{'#'*60}\n")
        
        # Step 1: Fetch data
        df = self.fetch_data(lookback_days)
        
        # Step 2: Engineer features
        df_features = self.engineer_features(df, threshold_percent, lookahead_periods)
        
        # Step 2.5: Feature Selection (NEW)
        selected_features = self.select_important_features(df_features, n_features=20)
        
        # Step 3: Preprocess data with selected features
        self.preprocess_data(df_features, selected_features=selected_features)
        
        # Step 4: Train base models (3-model ensemble)
        self.train_catboost()
        self.train_random_forest()
        self.train_logistic()
        
        # Step 4d: Train meta-learner (stacking)
        self.train_meta_learner()
        
        # Step 5: K-Fold Cross-Validation (if enabled)
        cv_results = None
        if run_cv:
            cv_results = self.run_kfold_validation(n_splits=5)
        
        # Step 6: Evaluate ensemble
        all_metrics = self.evaluate_ensemble()
        
        # Step 7: Save models
        model_dir = None
        if save_models:
            model_dir = self.save_models(
                metrics=all_metrics,
                threshold_percent=threshold_percent,
                lookahead_periods=lookahead_periods
            )
        
        # Check if we have validation set
        has_validation = 'validation' in all_metrics
        
        print(f"\n{'#'*60}")
        print(f"#  PIPELINE COMPLETE!")
        print(f"#  Symbol: {self.symbol} | Interval: {self.interval}")
        print(f"#")
        print(f"#  TRAINING SET:")
        print(f"#    Samples:   {len(self.prepared_data['y_train']) - self.prepared_data['X_train_lstm'].shape[1]}")
        print(f"#    Accuracy:  {all_metrics['train']['accuracy']*100:.2f}%")
        print(f"#    F1 Score:  {all_metrics['train']['f1_macro']:.4f}")
        print(f"#    ROC AUC:   {all_metrics['train']['roc_auc_ovr']:.4f}")
        print(f"#")
        if has_validation:
            print(f"#  VALIDATION SET:")
            print(f"#    Samples:   {len(self.prepared_data['y_val']) - self.prepared_data['X_val_lstm'].shape[1]}")
            print(f"#    Accuracy:  {all_metrics['validation']['accuracy']*100:.2f}%")
            print(f"#    F1 Score:  {all_metrics['validation']['f1_macro']:.4f}")
            print(f"#    ROC AUC:   {all_metrics['validation']['roc_auc_ovr']:.4f}")
            print(f"#")
        print(f"#  TEST SET:")
        print(f"#    Samples:   {len(self.prepared_data['y_test']) - self.prepared_data['X_test_lstm'].shape[1]}")
        print(f"#    Accuracy:  {all_metrics['test']['accuracy']*100:.2f}%")
        print(f"#    F1 Score:  {all_metrics['test']['f1_macro']:.4f}")
        print(f"#    ROC AUC:   {all_metrics['test']['roc_auc_ovr']:.4f}")
        print(f"#")
        print(f"#  OVERFITTING:")
        print(f"#    Train-Test Accuracy Gap: {all_metrics['overfitting_analysis']['train_test_accuracy_gap']*100:+.2f}%")
        print(f"#    Train-Test F1 Gap:       {all_metrics['overfitting_analysis']['train_test_f1_gap']:+.4f}")
        if model_dir:
            print(f"#")
            print(f"#  Models saved to: {model_dir}")
        print(f"{'#'*60}\n")
        
        return {
            'metrics': all_metrics,
            'model_dir': model_dir,
            'pipeline': self
        }


def main():
    """Main training script with regularization and cross-validation"""
    # Configuration
    SYMBOL = "BTCUSDT"
    INTERVAL = "1m"  # 1 minute intervals
    LOOKBACK = "35 days ago UTC"  # Increased to 35 days for 50,000+ data points (better ML training)
    THRESHOLD = 0.2  # 0.2% movement threshold (lowered from 0.5% for more signal on 1m candles)
    LOOKAHEAD = 6  # Look 6 periods ahead (6 minutes for 1m intervals)
    RUN_CV = True  # Enable k-fold cross-validation
    
    print("\n" + "="*60)
    print("🛡️  REGULARIZATION ENABLED:")
    print("  - L1/L2 Regularization")
    print("  - Reduced Model Complexity")
    print("  - Improved Scaling (RobustScaler)")
    print("  - K-Fold Cross-Validation (k=5)")
    print("="*60 + "\n")
    
    # Create and run pipeline
    pipeline = MLPipeline(symbol=SYMBOL, interval=INTERVAL)
    results = pipeline.run_full_pipeline(
        lookback_days=LOOKBACK,
        threshold_percent=THRESHOLD,
        lookahead_periods=LOOKAHEAD,
        save_models=True,
        run_cv=RUN_CV
    )
    
    print("Training complete!")
    print(f"Model directory: {results['model_dir']}")
    

if __name__ == "__main__":
    main()
