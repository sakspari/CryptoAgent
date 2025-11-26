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

    def train(self, X, y, selected_features, task="regression"):
        """Train the LightGBM model."""
        self.selected_features = selected_features
        X_final = X[selected_features].fillna(0)
        
        if task == "regression":
            self.model = lgb.LGBMRegressor(
                random_state=self.random_seed, 
                n_estimators=600, 
                learning_rate=0.05, 
                subsample=0.8, 
                colsample_bytree=0.8
            )
        else:
             self.model = lgb.LGBMClassifier(
                random_state=self.random_seed, 
                n_estimators=600, 
                learning_rate=0.05, 
                subsample=0.8, 
                colsample_bytree=0.8,
                class_weight="balanced"
            )
            
        self.model.fit(X_final, y)
        print(f"Model trained. Saving to {self.model_path}")
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
