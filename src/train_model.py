from src.utils.data_loader import batch_download, ASSETS, PERIOD_DAILY, PERIOD_WEEKLY, PERIOD_INTRADAY, DAILY_INTERVAL, WEEKLY_INTERVAL, INTRADAY_INTERVAL
from src.utils.feature_engineering import build_features_from_price
from src.utils.model_trainer import ModelTrainer
import pandas as pd
import yfinance as yf
import numpy as np

# Constants (should match notebook/utils)
EXOG = ["^VIX", "UUP", "GC=F", "^TNX"]
TARGET = "log_return_1d_ahead"
RANDOM_SEED = 42

def train():
    print("Starting Model Training Pipeline...")
    
    # 1. Download Data
    print("Downloading data...")
    daily = batch_download(ASSETS, PERIOD_DAILY, DAILY_INTERVAL)
    weekly = batch_download(ASSETS, PERIOD_WEEKLY, WEEKLY_INTERVAL)
    intra = batch_download(ASSETS, PERIOD_INTRADAY, INTRADAY_INTERVAL)
    
    # 2. Feature Engineering
    print("Building features...")
    features_df = build_features_from_price(daily, weekly, intra, assets=ASSETS)
    
    # Add Exogenous Features
    try:
        exog = yf.download(EXOG, period=PERIOD_DAILY, interval=DAILY_INTERVAL, group_by='column', progress=False)
        exog_close = {}
        for t in EXOG:
             if isinstance(exog.columns, pd.MultiIndex):
                exog_close[f"EXOG_{t.replace('^','').replace('=','')}"] = exog['Close'][t].rename(f"EXOG_{t}")
        
        if exog_close:
            exog_df = pd.concat(exog_close.values(), axis=1)
            features_df = features_df.join(exog_df)
    except Exception as e:
        print(f"Warning: Exogenous features failed: {e}")

    # 3. Prepare Target
    # Assuming we are predicting for BTC as in the notebook
    btc_close = daily['BTC-USD', 'Close'].rename("BTC_close")
    y_logret_ahead = np.log(btc_close).diff().shift(-1).rename("y_log_return_t+1")
    
    # Filter for target
    y = y_logret_ahead
    
    # Join and DropNA
    data = features_df.join(y).dropna()
    X = data.drop(columns=[y.name])
    y = data[y.name]
    
    # 4. Feature Selection
    # To allow the model to be reused for other assets (by renaming columns), 
    # we must train it ONLY on the features of the target asset (BTC) and shared EXOG features.
    # If we include ETH features in training, the model will expect them during prediction for other assets too.
    
    all_cols = X.columns.tolist()
    selected_features = [c for c in all_cols if "BTC-USD" in c or "EXOG" in c]
    
    print(f"Selected {len(selected_features)} features (BTC + EXOG) for generic training.") 
    
    # 5. Train Model
    print(f"Training model with {len(selected_features)} features...")
    trainer = ModelTrainer()
    trainer.train(X, y, selected_features, task="regression")
    print("Training complete.")

if __name__ == "__main__":
    train()
