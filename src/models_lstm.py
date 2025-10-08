"""
LSTM-Based Models Module with Lasso Meta-Learner
Implements temporal sequence modeling for Bitcoin price prediction

Architecture:
  Base Model: LSTM/GRU (learns from 30-candle sequences)
  Meta-Learner: Lasso Regression (L1 regularization for feature selection)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import os


class LSTMModel:
    """LSTM model for temporal sequence prediction"""
    
    def __init__(self, n_classes=2, sequence_length=30, n_features=20):
        """
        Initialize LSTM model
        
        Args:
            n_classes: Number of output classes (2 for binary: Down/Up)
            sequence_length: Number of historical candles (default 30)
            n_features: Number of features per candle (default 20)
        """
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build(self, lstm_units=64, dropout=0.3, l2_reg=0.01, learning_rate=0.001):
        """
        Build LSTM architecture with regularization
        
        Args:
            lstm_units: Number of LSTM units (default 64)
            dropout: Dropout rate for regularization (default 0.3)
            l2_reg: L2 regularization strength (default 0.01)
            learning_rate: Adam optimizer learning rate (default 0.001)
        """
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # LSTM layer 1 with L2 regularization
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # LSTM layer 2 with L2 regularization
            layers.LSTM(
                lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # Dense layer with L2 regularization
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.Dropout(dropout),
            
            # Output layer - Binary: 1 unit with sigmoid, Multi-class: n_classes with softmax
            layers.Dense(
                1 if self.n_classes == 2 else self.n_classes,
                activation='sigmoid' if self.n_classes == 2 else 'softmax'
            )
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.n_classes == 2:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
            )
        
        self.model = model
        print(f"✓ LSTM model built")
        print(f"  - Input shape: ({self.sequence_length}, {self.n_features})")
        print(f"  - LSTM units: {lstm_units} -> {lstm_units // 2}")
        print(f"  - Dropout: {dropout}")
        print(f"  - L2 regularization: {l2_reg}")
        print(f"  - Output classes: {self.n_classes}")
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, early_stopping_patience=10):
        """
        Train LSTM model with early stopping
        
        Args:
            X_train: Training sequences (samples, sequence_length, n_features)
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Maximum training epochs (default 50)
            batch_size: Batch size (default 32)
            early_stopping_patience: Early stopping patience (default 10)
        """
        print("\nTraining LSTM...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Input shape: {X_train.shape}")
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"  Validation samples: {len(X_val)}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Print final metrics
        final_epoch = len(history.history['loss'])
        train_acc = history.history['accuracy'][-1]
        print(f"\n✓ LSTM training complete")
        print(f"  - Epochs trained: {final_epoch}/{epochs}")
        print(f"  - Final training accuracy: {train_acc:.4f}")
        
        if validation_data:
            val_acc = history.history['val_accuracy'][-1]
            print(f"  - Final validation accuracy: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.model.predict(X, verbose=0)
        if self.n_classes == 2:
            # Binary: output shape is (batch_size, 1)
            return (proba.flatten() > 0.5).astype(int)
        else:
            # Multi-class: output shape is (batch_size, n_classes)
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        proba = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Binary: output is (batch_size, 1) with P(class=1)
            # Return [P(class=0), P(class=1)]
            proba = proba.flatten()
            return np.column_stack([1 - proba, proba])
        else:
            # Multi-class: already in correct format
            return proba
    
    def save(self, filepath):
        """Save LSTM model to file"""
        self.model.save(filepath)
        print(f"✓ LSTM model saved to {filepath}")
    
    def load(self, filepath):
        """Load LSTM model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ LSTM model loaded from {filepath}")
        return self


class GRUModel:
    """GRU model for temporal sequence prediction (faster alternative to LSTM)"""
    
    def __init__(self, n_classes=2, sequence_length=30, n_features=20):
        """
        Initialize GRU model
        
        Args:
            n_classes: Number of output classes (2 for binary: Down/Up)
            sequence_length: Number of historical candles (default 30)
            n_features: Number of features per candle (default 20)
        """
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build(self, gru_units=64, dropout=0.3, l2_reg=0.01, learning_rate=0.001):
        """
        Build GRU architecture with regularization
        
        Args:
            gru_units: Number of GRU units (default 64)
            dropout: Dropout rate for regularization (default 0.3)
            l2_reg: L2 regularization strength (default 0.01)
            learning_rate: Adam optimizer learning rate (default 0.001)
        """
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # GRU layer 1 with L2 regularization
            layers.GRU(
                gru_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # GRU layer 2 with L2 regularization
            layers.GRU(
                gru_units // 2,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg)
            ),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            
            # Dense layer with L2 regularization
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg)
            ),
            layers.Dropout(dropout),
            
            # Output layer - Binary: 1 unit with sigmoid, Multi-class: n_classes with softmax
            layers.Dense(
                1 if self.n_classes == 2 else self.n_classes,
                activation='sigmoid' if self.n_classes == 2 else 'softmax'
            )
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.n_classes == 2:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc', multi_label=True)]
            )
        
        self.model = model
        print(f"✓ GRU model built")
        print(f"  - Input shape: ({self.sequence_length}, {self.n_features})")
        print(f"  - GRU units: {gru_units} -> {gru_units // 2}")
        print(f"  - Dropout: {dropout}")
        print(f"  - L2 regularization: {l2_reg}")
        print(f"  - Output classes: {self.n_classes}")
        
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, early_stopping_patience=10):
        """Train GRU model with early stopping (same as LSTM)"""
        print("\nTraining GRU...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Input shape: {X_train.shape}")
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"  Validation samples: {len(X_val)}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Print final metrics
        final_epoch = len(history.history['loss'])
        train_acc = history.history['accuracy'][-1]
        print(f"\n✓ GRU training complete")
        print(f"  - Epochs trained: {final_epoch}/{epochs}")
        print(f"  - Final training accuracy: {train_acc:.4f}")
        
        if validation_data:
            val_acc = history.history['val_accuracy'][-1]
            print(f"  - Final validation accuracy: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.model.predict(X, verbose=0)
        if self.n_classes == 2:
            # Binary: output shape is (batch_size, 1)
            return (proba.flatten() > 0.5).astype(int)
        else:
            # Multi-class: output shape is (batch_size, n_classes)
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        proba = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Binary: output is (batch_size, 1) with P(class=1)
            # Return [P(class=0), P(class=1)]
            proba = proba.flatten()
            return np.column_stack([1 - proba, proba])
        else:
            # Multi-class: already in correct format
            return proba
    
    def save(self, filepath):
        """Save GRU model to file"""
        self.model.save(filepath)
        print(f"✓ GRU model saved to {filepath}")
    
    def load(self, filepath):
        """Load GRU model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ GRU model loaded from {filepath}")
        return self


class LSTMEnsemble:
    """
    Ensemble model with LSTM base learner + ElasticNet meta-learner
    
    Architecture:
      1. LSTM/GRU processes 30-candle sequences
      2. ElasticNet Regression (L1 + L2 regularization) acts as meta-learner
         - Learns from LSTM's probability outputs
         - Combines L1 (feature selection) + L2 (stability)
         - More robust than pure Lasso (won't zero everything)
         - Interpretable coefficients
    """
    
    def __init__(self, n_classes=2, sequence_length=30, n_features=20, use_gru=False):
        """
        Initialize LSTM ensemble with ElasticNet meta-learner
        
        Args:
            n_classes: Number of output classes (2 for binary)
            sequence_length: Number of historical candles (30)
            n_features: Number of features per candle (20)
            use_gru: Use GRU instead of LSTM (faster, slightly less accurate)
        """
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.use_gru = use_gru
        
        # Base model (LSTM or GRU)
        if use_gru:
            self.base_model = GRUModel(n_classes, sequence_length, n_features)
        else:
            self.base_model = LSTMModel(n_classes, sequence_length, n_features)
        
        # Meta-learner (ElasticNet Regression)
        self.meta_learner = None
        self.use_stacking = False
        
    def build(self, lstm_units=64, dropout=0.3, l2_reg=0.01, learning_rate=0.001):
        """Build LSTM/GRU base model"""
        self.base_model.build(
            lstm_units=lstm_units if not self.use_gru else lstm_units,
            dropout=dropout,
            l2_reg=l2_reg,
            learning_rate=learning_rate
        )
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, early_stopping_patience=10):
        """Train base LSTM/GRU model"""
        history = self.base_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience
        )
        return history
    
    def train_meta_learner(self, X_train, y_train, alpha=0.01, l1_ratio=0.5):
        """
        Train ElasticNet meta-learner on LSTM outputs
        
        Args:
            X_train: Training sequences (for generating LSTM predictions)
            y_train: Training labels
            alpha: Overall regularization strength (default 0.01)
            l1_ratio: Balance between L1 and L2 (default 0.5)
                     - 0.0 = Pure Ridge (L2 only)
                     - 0.5 = Balanced (50% L1, 50% L2) - RECOMMENDED
                     - 1.0 = Pure Lasso (L1 only)
        """
        print("\n" + "="*60)
        print("TRAINING ELASTICNET META-LEARNER")
        print("="*60)
        
        # Get LSTM probability outputs
        print("Generating base model predictions...")
        base_proba = self.base_model.predict_proba(X_train)
        
        print(f"  Base model output shape: {base_proba.shape}")
        print(f"  Number of meta-features: {base_proba.shape[1]}")
        
        # Train ElasticNet on LSTM outputs
        print(f"\nTraining ElasticNet regression...")
        print(f"  Alpha (overall strength): {alpha}")
        print(f"  L1 ratio: {l1_ratio} ({l1_ratio*100:.0f}% L1, {(1-l1_ratio)*100:.0f}% L2)")
        
        self.meta_learner = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=42,
            selection='cyclic'
        )
        
        self.meta_learner.fit(base_proba, y_train)
        self.use_stacking = True
        
        # Analyze coefficients
        coefficients = self.meta_learner.coef_
        non_zero_features = np.sum(coefficients != 0)
        
        print(f"\n✓ ElasticNet meta-learner trained")
        print(f"  - Overall penalty (alpha): {alpha}")
        print(f"  - L1/L2 ratio: {l1_ratio}")
        print(f"  - Features with non-zero coefficients: {non_zero_features}/{len(coefficients)}")
        print(f"  - Intercept: {self.meta_learner.intercept_:.4f}")
        
        # Show coefficients
        print(f"\n  Coefficients:")
        for i, coef in enumerate(coefficients):
            class_name = "Down" if i == 0 else "Up"
            status = "✓ ACTIVE" if coef != 0 else "✗ ZEROED"
            print(f"    P({class_name}): {coef:>8.4f}  {status}")
        
        # Evaluate meta-learner
        meta_pred = self.meta_learner.predict(base_proba)
        meta_pred_binary = (meta_pred > 0.5).astype(int)
        
        train_acc = accuracy_score(y_train, meta_pred_binary)
        train_f1 = f1_score(y_train, meta_pred_binary, average='macro')
        
        print(f"\n  Training performance:")
        print(f"    Accuracy: {train_acc:.4f}")
        print(f"    F1 Score: {train_f1:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels using meta-learner if available"""
        if self.use_stacking and self.meta_learner is not None:
            # Use meta-learner
            base_proba = self.base_model.predict_proba(X)
            meta_pred = self.meta_learner.predict(base_proba)
            return (meta_pred > 0.5).astype(int)
        else:
            # Use base model only
            return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.use_stacking and self.meta_learner is not None:
            # Get base model probabilities
            base_proba = self.base_model.predict_proba(X)
            
            # Meta-learner outputs continuous values
            # Convert to probabilities
            meta_pred = self.meta_learner.predict(base_proba)
            meta_pred = np.clip(meta_pred, 0, 1)  # Clip to [0, 1]
            
            # Return as [P(Down), P(Up)]
            return np.column_stack([1 - meta_pred, meta_pred])
        else:
            # Use base model probabilities
            return self.base_model.predict_proba(X)
    
    def save(self, model_dir):
        """Save ensemble to directory"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save base model
        model_type = "gru" if self.use_gru else "lstm"
        base_path = os.path.join(model_dir, f"{model_type}_model.keras")
        self.base_model.save(base_path)
        
        # Save meta-learner if trained
        if self.meta_learner is not None:
            meta_path = os.path.join(model_dir, "elasticnet_meta_learner.pkl")
            joblib.dump(self.meta_learner, meta_path)
            print(f"✓ ElasticNet meta-learner saved to {meta_path}")
        
        # Save metadata
        metadata = {
            'n_classes': self.n_classes,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'use_gru': self.use_gru,
            'use_stacking': self.use_stacking,
            'model_type': model_type
        }
        
        metadata_path = os.path.join(model_dir, "lstm_ensemble_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ LSTM ensemble metadata saved to {metadata_path}")
        
    def load(self, model_dir):
        """Load ensemble from directory"""
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "lstm_ensemble_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load base model
        model_type = metadata['model_type']
        base_path = os.path.join(model_dir, f"{model_type}_model.keras")
        self.base_model.load(base_path)
        
        # Load meta-learner if exists
        meta_path = os.path.join(model_dir, "elasticnet_meta_learner.pkl")
        if os.path.exists(meta_path):
            self.meta_learner = joblib.load(meta_path)
            self.use_stacking = True
            print(f"✓ ElasticNet meta-learner loaded from {meta_path}")
        else:
            # Try legacy Lasso path for backwards compatibility
            legacy_path = os.path.join(model_dir, "lasso_meta_learner.pkl")
            if os.path.exists(legacy_path):
                self.meta_learner = joblib.load(legacy_path)
                self.use_stacking = True
                print(f"✓ Legacy Lasso meta-learner loaded from {legacy_path}")
        
        print(f"✓ LSTM ensemble loaded from {model_dir}")
        
        return self
