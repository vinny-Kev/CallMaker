"""
Transformer Regressor V2 - SHARPENED FOR TRADING
- MAE loss (less conservative than MSE)
- Huber loss (robust to outliers)
- Prediction calibration
- Confidence estimation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
from datetime import datetime


class PositionalEncoding(layers.Layer):
    """Add positional encoding to input sequences"""
    
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
        
    def get_config(self):
        config = super().get_config()
        return config
        
    def positional_encoding(self, sequence_length, d_model):
        """Generate positional encodings"""
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((sequence_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def get_config(self):
        config = super().get_config()
        return config
        
    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerRegressorV2:
    """Enhanced Transformer regressor with multiple loss functions"""
    
    def __init__(self, sequence_length=60, n_features=30, d_model=64, 
                 num_heads=4, num_blocks=2, ff_dim=128, dropout=0.1,
                 loss_type='mae'):  # 'mae', 'mse', 'huber', 'logcosh'
        """
        Args:
            loss_type: 
                - 'mae': Mean Absolute Error (less conservative, better for trading)
                - 'mse': Mean Squared Error (conservative, penalizes big errors)
                - 'huber': Huber loss (balanced, robust to outliers)
                - 'logcosh': Log-cosh (smooth approximation of MAE)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.loss_type = loss_type
        self.model = None
        self.history = None
        
    def build(self):
        """Build the Transformer model"""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Project input features to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.d_model)(x)
        
        # Stack transformer blocks
        for _ in range(self.num_blocks):
            x = TransformerBlock(self.d_model, self.num_heads, self.ff_dim, self.dropout)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        # REGRESSION OUTPUT: Linear activation
        outputs = layers.Dense(1, activation='linear', name='regression_output')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='transformer_regressor_v2')
        
        # Select loss function
        if self.loss_type == 'mae':
            loss = 'mae'
            metrics = ['mae', 'mse']
        elif self.loss_type == 'mse':
            loss = 'mse'
            metrics = ['mae', 'mse']
        elif self.loss_type == 'huber':
            loss = keras.losses.Huber(delta=1.0)  # delta=1.0 is good for ~1% moves
            metrics = ['mae', 'mse']
        elif self.loss_type == 'logcosh':
            # LogCosh is not a built-in string, use custom loss
            def log_cosh_loss(y_true, y_pred):
                return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))
            loss = log_cosh_loss
            metrics = ['mae', 'mse']
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        print(f"\n✓ Using {self.loss_type.upper()} loss function")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        # Build model if not already built
        if self.model is None:
            self.build()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"\n{'='*60}")
        print(f"TRAINING TRANSFORMER REGRESSOR V2 ({self.loss_type.upper()} LOSS)")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Target (y_train) - Mean: {y_train.mean():.3f}%, Std: {y_train.std():.3f}%")
        print(f"Target (y_train) - Min: {y_train.min():.3f}%, Max: {y_train.max():.3f}%")
        print(f"Target (y_val) - Mean: {y_val.mean():.3f}%, Std: {y_val.std():.3f}%")
        print(f"{'='*60}\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Predict price changes"""
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_with_confidence(self, X, n_samples=10):
        """
        Predict with confidence estimation using Monte Carlo Dropout
        
        Returns:
            predictions: Mean predictions
            uncertainty: Standard deviation of predictions (confidence)
        """
        predictions = []
        
        # Enable dropout during inference
        for _ in range(n_samples):
            pred = self.model(X, training=True)  # Keep dropout active
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def calibrate_predictions(self, X_train, y_train, X_val, y_val):
        """
        Calibrate predictions to match actual distribution
        
        Returns:
            scale_factor: Multiply predictions by this to match actual std
        """
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)
        
        # Calculate standard deviations
        actual_train_std = y_train.std()
        actual_val_std = y_val.std()
        pred_train_std = train_pred.std()
        pred_val_std = val_pred.std()
        
        # Scale factor to match actual volatility
        scale_factor = (actual_train_std + actual_val_std) / (pred_train_std + pred_val_std + 1e-10)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION CALIBRATION")
        print(f"{'='*60}")
        print(f"Actual train std: {actual_train_std:.3f}%")
        print(f"Predicted train std: {pred_train_std:.3f}%")
        print(f"Actual val std: {actual_val_std:.3f}%")
        print(f"Predicted val std: {pred_val_std:.3f}%")
        print(f"Scale factor: {scale_factor:.3f}x")
        print(f"{'='*60}\n")
        
        return scale_factor
    
    def save_model(self, model_dir):
        """Save model and metadata"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(model_dir, 'transformer_regressor_v2_model.keras')
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_type': 'transformer_regressor_v2',
            'loss_type': self.loss_type,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(model_dir, 'transformer_regressor_v2_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    @classmethod
    def load_model(cls, model_dir):
        """Load model and metadata"""
        metadata_path = os.path.join(model_dir, 'transformer_regressor_v2_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            sequence_length=metadata['sequence_length'],
            n_features=metadata['n_features'],
            d_model=metadata['d_model'],
            num_heads=metadata['num_heads'],
            num_blocks=metadata['num_blocks'],
            ff_dim=metadata['ff_dim'],
            dropout=metadata['dropout'],
            loss_type=metadata.get('loss_type', 'mae')
        )
        
        # Load model
        model_path = os.path.join(model_dir, 'transformer_regressor_v2_model.keras')
        instance.model = keras.models.load_model(
            model_path,
            custom_objects={
                'PositionalEncoding': PositionalEncoding,
                'TransformerBlock': TransformerBlock
            }
        )
        
        return instance
