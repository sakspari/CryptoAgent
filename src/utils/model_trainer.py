import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class ModelTrainer:
    def __init__(self, model_path="artifacts/final_lgbm_model.pkl", random_seed=42):
        self.model_path = model_path
        self.random_seed = random_seed
        self.model = None
        self.selected_features = [] # Should be loaded or defined

    def train(self, X, y, feature_names=None, task="regression", save_model=True):
        """
        Trains a LightGBM model.
        """
        if feature_names is None:
            feature_names = X.columns.tolist()
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Dataset
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
        
        # Params
        params = {
            "objective": task,
            "metric": "rmse" if task == "regression" else "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 15, # Reduced from 31 for small dataset
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10, # Reduced from default 20
            "verbosity": -1 # Suppress warnings
        }
        
        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        self.selected_features = feature_names # Changed from self.features to self.selected_features to match class attribute
        
        if save_model:
            print(f"Model trained. Saving to {self.model_path}") # Corrected syntax
            self.save_model()

    def save_model(self):
        """Save the trained model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        # Also save selected features if possible, or assume they are fixed/passed in
        joblib.dump(self.selected_features, self.model_path.replace(".pkl", "_features.pkl"))

    def load_model(self):
        """Load the trained model."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            feat_path = self.model_path.replace(".pkl", "_features.pkl")
            if os.path.exists(feat_path):
                self.selected_features = joblib.load(feat_path)
            return True
        return False

    def predict_next_day(self, latest_features_df):
        """Predict for the next day using the latest features."""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or found.")
        
        # Ensure we have the right features
        if not self.selected_features:
             # Fallback if features not saved, might fail if mismatch
             self.selected_features = latest_features_df.columns.tolist()

        X_predict = latest_features_df[self.selected_features].fillna(0)
        
        if X_predict.empty:
            return None
            
        # Take the last row (latest data)
        latest_row = X_predict.iloc[[-1]]
        prediction = self.model.predict(latest_row)
        return prediction[0]
