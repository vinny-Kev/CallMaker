"""
Machine Learning Models Module
Contains CatBoost, Random Forest, and Logistic Regression models for Bitcoin prediction
No LSTM - removed for better performance and reliability
Includes Stacking Meta-Learner for improved ensemble performance
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import os


class CatBoostModel:
    """CatBoost Gradient Boosting model"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, iterations=500, learning_rate=0.1, depth=6, l2_leaf_reg=3, 
              random_strength=1, bagging_temperature=1, class_weights=None):
        """Build CatBoost model with regularization"""
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            verbose=False,
            random_seed=42,
            class_weights=class_weights,
            loss_function='MultiClass' if self.n_classes > 2 else 'Logloss'
        )
        
        print("✓ CatBoost model built")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model"""
        print("\nTraining CatBoost...")
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=50 if eval_set else None
        )
        
        print("✓ CatBoost training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = self.model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model"""
        self.model.save_model(path)
        print(f"✓ CatBoost model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        print(f"✓ CatBoost model loaded from {path}")


class RandomForestModel:
    """Random Forest ensemble model"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        
    def build(self, class_weights=None, n_estimators=50, max_depth=8, 
                    min_samples_split=20, min_samples_leaf=10, max_features='sqrt'):
        """Build Random Forest model with reduced complexity to prevent overfitting"""
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,        # Reduced from 200 to 50
            max_depth=max_depth,              # Reduced from 15 to 8  
            min_samples_split=min_samples_split,  # Increased from 10 to 20
            min_samples_leaf=min_samples_leaf,    # Increased from 5 to 10
            max_features=max_features,            # Keep sqrt for regularization
            class_weight='balanced' if class_weights else None,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("✓ Random Forest model built with reduced complexity")
        print(f"  Architecture: {n_estimators} trees, max_depth={max_depth}, regularization enhanced")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        self.model.fit(X_train, y_train)
        
        print("✓ Random Forest training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, path)
        print(f"✓ Random Forest model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = joblib.load(path)
        print(f"✓ Random Forest model loaded from {path}")


class LogisticRegressionModel:
    """Logistic Regression model for baseline and ensemble balance"""
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.model = None
        self.scaler = StandardScaler()
        
    def build(self, class_weights=True, C=1.0, max_iter=1000, solver='lbfgs'):
        """Build Logistic Regression model"""
        
        self.model = LogisticRegression(
            C=C,                              # Regularization strength (lower = more regularization)
            max_iter=max_iter,
            solver=solver,                    # lbfgs works well for small datasets
            class_weight='balanced' if class_weights else None,
            random_state=42,
            multi_class='ovr'                 # One-vs-rest for multi-class
        )
        
        print("✓ Logistic Regression model built")
        print(f"  Regularization: C={C}, solver={solver}")
        return self
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Logistic Regression model with feature scaling"""
        print("\nTraining Logistic Regression...")
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        
        print("✓ Logistic Regression training complete")
        return self
    
    def predict(self, X):
        """Get predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance (coefficients)"""
        if self.n_classes == 2:
            coeffs = self.model.coef_[0]
        else:
            # For multi-class, use average absolute coefficients
            coeffs = np.mean(np.abs(self.model.coef_), axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)
        return feature_importance
    
    def save(self, path):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        print(f"✓ Logistic Regression model saved to {path}")
    
    def load(self, path):
        """Load model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        print(f"✓ Logistic Regression model loaded from {path}")


class EnsembleModel:
    """
    Stacked Ensemble with Meta-Learner
    - Base learners: CatBoost, Random Forest, Logistic Regression
    - Meta-learner: Logistic Regression (trained on base predictions)
    """
    
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.catboost = CatBoostModel(n_classes)
        self.random_forest = RandomForestModel(n_classes)
        self.logistic = LogisticRegressionModel(n_classes)
        self.meta_learner = None  # Will be trained on base predictions
        self.weights = None  # Fallback for weighted averaging if meta-learner not trained
        self.use_stacking = False  # Flag to determine prediction method
        
    def train_meta_learner(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train meta-learner on base model predictions
        
        Args:
            X_train: Original training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        print("\n" + "="*60)
        print("TRAINING META-LEARNER (Stacking)")
        print("="*60)
        
        # Generate base predictions for training set
        print("Generating base model predictions for training data...")
        catboost_proba = self.catboost.predict_proba(X_train)
        rf_proba = self.random_forest.predict_proba(X_train)
        logistic_proba = self.logistic.predict_proba(X_train)
        
        # Stack predictions: shape (n_samples, n_models * n_classes)
        # Each base model outputs 3 probabilities (one per class)
        X_meta_train = np.hstack([catboost_proba, rf_proba, logistic_proba])
        print(f"  Meta-features shape: {X_meta_train.shape}")
        print(f"  (Each sample has {self.n_classes} probs × 3 models = {X_meta_train.shape[1]} features)")
        
        # Initialize meta-learner (Logistic Regression with stronger regularization)
        from sklearn.linear_model import LogisticRegression
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
            class_weight='balanced',
            C=0.1,  # Stronger regularization (smaller C = more regularization)
            penalty='l2'  # L2 regularization
        )
        
        # Train meta-learner
        print("\nTraining meta-learner on base predictions...")
        self.meta_learner.fit(X_meta_train, y_train)
        self.use_stacking = True
        
        # Evaluate on training set
        meta_pred_train = self.meta_learner.predict(X_meta_train)
        train_acc = accuracy_score(y_train, meta_pred_train)
        train_f1 = f1_score(y_train, meta_pred_train, average='macro')
        print(f"  Meta-learner Train Accuracy: {train_acc:.4f}")
        print(f"  Meta-learner Train F1: {train_f1:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("\nEvaluating meta-learner on validation data...")
            catboost_proba_val = self.catboost.predict_proba(X_val)
            rf_proba_val = self.random_forest.predict_proba(X_val)
            logistic_proba_val = self.logistic.predict_proba(X_val)
            
            X_meta_val = np.hstack([catboost_proba_val, rf_proba_val, logistic_proba_val])
            meta_pred_val = self.meta_learner.predict(X_meta_val)
            
            val_acc = accuracy_score(y_val, meta_pred_val)
            val_f1 = f1_score(y_val, meta_pred_val, average='macro')
            print(f"  Meta-learner Val Accuracy: {val_acc:.4f}")
            print(f"  Meta-learner Val F1: {val_f1:.4f}")
        
        print("\n✓ Meta-learner training complete!")
        print("  Stacking enabled: Base models → Meta-learner → Final prediction")
        
    def set_weights(self, catboost_weight=0.5, rf_weight=0.25, logistic_weight=0.25):
        """Set ensemble weights - 3 model ensemble"""
        total = catboost_weight + rf_weight + logistic_weight
        self.weights = {
            'catboost': catboost_weight / total,
            'rf': rf_weight / total,
            'logistic': logistic_weight / total
        }
        print(f"✓ Ensemble weights set: {self.weights}")
        print(f"  CatBoost: {catboost_weight/total:.1%}, RF: {rf_weight/total:.1%}, Logistic: {logistic_weight/total:.1%}")
    
    def predict_proba(self, X_regular):
        """
        Get ensemble prediction probabilities
        Uses stacking meta-learner if trained, otherwise falls back to weighted averaging
        
        Args:
            X_regular: Features for CatBoost, Random Forest, and Logistic Regression
            
        Returns:
            Final prediction probabilities (either from meta-learner or weighted average)
        """
        # Get base model predictions
        catboost_proba = self.catboost.predict_proba(X_regular)
        rf_proba = self.random_forest.predict_proba(X_regular)
        logistic_proba = self.logistic.predict_proba(X_regular)
        
        # Use stacking if meta-learner is trained
        if self.use_stacking and self.meta_learner is not None:
            # Stack base predictions as meta-features
            X_meta = np.hstack([catboost_proba, rf_proba, logistic_proba])
            
            # Meta-learner makes final prediction
            return self.meta_learner.predict_proba(X_meta)
        else:
            # Fallback to weighted averaging
            if self.weights is None:
                self.set_weights()
            
            ensemble_proba = (
                self.weights['catboost'] * catboost_proba +
                self.weights['rf'] * rf_proba +
                self.weights['logistic'] * logistic_proba
            )
            
            return ensemble_proba
    
    def predict(self, X_regular):
        """Get ensemble predictions"""
        proba = self.predict_proba(X_regular)
        return np.argmax(proba, axis=1)
    
    def save(self, save_dir):
        """Save all models including meta-learner"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save base models
        self.catboost.save(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.save(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.logistic.save(os.path.join(save_dir, 'logistic_model.pkl'))
        
        # Save meta-learner if trained
        if self.meta_learner is not None:
            meta_path = os.path.join(save_dir, 'meta_learner.pkl')
            joblib.dump(self.meta_learner, meta_path)
            print(f"✓ Meta-learner saved to {meta_path}")
        
        # Save ensemble config
        ensemble_config = {
            'weights': self.weights,
            'use_stacking': self.use_stacking,
            'has_meta_learner': self.meta_learner is not None
        }
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(ensemble_config, f)
        
        print(f"✓ Ensemble models saved to {save_dir}")
        if self.use_stacking:
            print(f"  Stacking: ENABLED (meta-learner trained)")
        else:
            print(f"  Stacking: DISABLED (using weighted averaging)")
    
    def load(self, save_dir):
        """Load all models including meta-learner"""
        self.catboost.load(os.path.join(save_dir, 'catboost_model.cbm'))
        self.random_forest.load(os.path.join(save_dir, 'random_forest_model.pkl'))
        self.logistic.load(os.path.join(save_dir, 'logistic_model.pkl'))
        
        # Load ensemble config
        with open(os.path.join(save_dir, 'ensemble_weights.json'), 'r') as f:
            ensemble_config = json.load(f)
        
        # Handle backward compatibility
        if isinstance(ensemble_config, dict) and 'use_stacking' in ensemble_config:
            self.weights = ensemble_config.get('weights')
            self.use_stacking = ensemble_config.get('use_stacking', False)
            
            # Load meta-learner if it exists
            meta_path = os.path.join(save_dir, 'meta_learner.pkl')
            if ensemble_config.get('has_meta_learner') and os.path.exists(meta_path):
                self.meta_learner = joblib.load(meta_path)
                print(f"✓ Meta-learner loaded from {meta_path}")
        else:
            # Old format: just weights
            self.weights = ensemble_config
            self.use_stacking = False
            self.meta_learner = None
        
        print(f"✓ Ensemble models loaded from {save_dir}")
        print(f"  Weights: {self.weights}")
        if self.use_stacking:
            print(f"  Stacking: ENABLED")
        else:
            print(f"  Stacking: DISABLED (using weighted averaging)")