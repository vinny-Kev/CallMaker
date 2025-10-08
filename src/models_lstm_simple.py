"""
Simple LSTM Model for Bitcoin Price Prediction
Single LSTM model without meta-learner complexity

Architecture:
  LSTM (2 layers) → Dense → Binary Classification
  No ensemble, no meta-learner - just clean, direct predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import joblib
import json
import os


class SimpleLSTMModel:
    """Simple LSTM model for temporal sequence prediction"""
    
    def __init__(self, sequence_length=60, n_features=20):
        """
        Initialize Simple LSTM model
        
        Args:
            sequence_length: Number of historical candles (default 60)
            n_features: Number of features per candle (default 20)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build(self, lstm_units=64, dropout=0.3, l2_reg=0.01, learning_rate=0.001):
        """
        Build LSTM architecture with regularization
        
        Args:
            lstm_units: Number of LSTM units in first layer (default 64)
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
            
            # Output layer - Binary classification with sigmoid
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        
        print(f"\n✓ LSTM model built")
        print(f"  - Input shape: ({self.sequence_length}, {self.n_features})")
        print(f"  - LSTM units: {lstm_units} -> {lstm_units // 2}")
        print(f"  - Dropout: {dropout}")
        print(f"  - L2 regularization: {l2_reg}")
        print(f"  - Binary classification: 0=Down, 1=Up")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
             epochs=50, batch_size=32, early_stopping_patience=10,
             class_weights=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, n_features)
            y_train: Training labels (samples,)
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            early_stopping_patience: Patience for early stopping
            class_weights: Class weights for imbalanced data
            
        Returns:
            Training history
        """
        print(f"\nTraining LSTM...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Input shape: {X_train.shape}")
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='loss' if X_val is None else 'val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss' if X_val is None else 'val_loss',
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
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get final training accuracy
        train_pred = (self.model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
        train_acc = (train_pred == y_train).mean()
        
        print(f"\n✓ LSTM training complete")
        print(f"  - Epochs trained: {len(history.history['loss'])}/{epochs}")
        print(f"  - Final training accuracy: {train_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Predict class labels (0=Down, 1=Up)
        
        Args:
            X: Input sequences (samples, sequence_length, n_features)
            
        Returns:
            Predicted class labels (samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Get probabilities and threshold at 0.5
        proba = self.model.predict(X, verbose=0)
        predictions = (proba > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities [P(Down), P(Up)]
        
        Args:
            X: Input sequences (samples, sequence_length, n_features)
            
        Returns:
            Probability matrix (samples, 2) with [P(Down), P(Up)]
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Get P(Up) from sigmoid output
        p_up = self.model.predict(X, verbose=0).flatten()
        
        # Calculate P(Down) = 1 - P(Up)
        p_down = 1 - p_up
        
        # Stack into [P(Down), P(Up)] format
        proba = np.column_stack([p_down, p_up])
        
        return proba
    
    def save(self, model_dir):
        """
        Save LSTM model and metadata
        
        Args:
            model_dir: Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(model_dir, 'lstm_model.keras')
        self.model.save(model_path)
        print(f"✓ LSTM model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'model_type': 'SimpleLSTM',
            'architecture': 'LSTM (2-layer) → Dense → Binary'
        }
        
        metadata_path = os.path.join(model_dir, 'lstm_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ LSTM metadata saved to {metadata_path}")
        
        return model_dir
    
    @classmethod
    def load(cls, model_dir):
        """
        Load LSTM model from directory
        
        Args:
            model_dir: Directory containing saved model files
            
        Returns:
            Loaded SimpleLSTMModel instance
        """
        # Load metadata
        metadata_path = os.path.join(model_dir, 'lstm_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            sequence_length=metadata['sequence_length'],
            n_features=metadata['n_features']
        )
        
        # Load Keras model
        model_path = os.path.join(model_dir, 'lstm_model.keras')
        instance.model = keras.models.load_model(model_path)
        
        print(f"✓ Loaded SimpleLSTM model from {model_dir}")
        print(f"  - Sequence length: {instance.sequence_length}")
        print(f"  - Features: {instance.n_features}")
        
        return instance
