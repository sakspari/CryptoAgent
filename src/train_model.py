from src.utils.data_loader import batch_download, PERIOD_DAILY, PERIOD_WEEKLY, PERIOD_INTRADAY, DAILY_INTERVAL, WEEKLY_INTERVAL, INTRADAY_INTERVAL
from src.utils.feature_engineering import build_features_from_price
from src.utils.model_trainer import ModelTrainer
import pandas as pd
import yfinance as yf
import numpy as np
import os

# Constants
EXOG = ["^VIX", "UUP", "GC=F", "^TNX"]

def train_and_predict(ticker: str):
    """
    Downloads data, trains a model, and predicts for a specific ticker.
    """
    print(f"Starting Dynamic Analysis for {ticker}...")
    assets = [ticker]
    
    # 1. Download Data
    print(f"Downloading data for {ticker}...")
    try:
        daily = batch_download(assets, PERIOD_DAILY, DAILY_INTERVAL)
        weekly = batch_download(assets, PERIOD_WEEKLY, WEEKLY_INTERVAL)
        intra = batch_download(assets, PERIOD_INTRADAY, INTRADAY_INTERVAL)
    except Exception as e:
        return {"error": f"Data download failed: {e}"}
    
    # 2. Feature Engineering
    print("Building features...")
    try:
        features_df = build_features_from_price(daily, weekly, intra, assets=assets)
    except Exception as e:
        return {"error": f"Feature engineering failed: {e}"}
    
    # Add Exogenous Features
    try:
        exog = yf.download(EXOG, period=PERIOD_DAILY, interval=DAILY_INTERVAL, group_by='column', progress=False, auto_adjust=False)
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
    # We predict for the specific ticker
    try:
        # Ticker column name might vary depending on how data_loader normalized it
        # But build_features renames them to {ticker}_close
        target_col = f"{ticker}_close"
        if target_col not in features_df.columns:
             # Try finding the close column
             cols = [c for c in features_df.columns if "close" in c and ticker in c]
             if cols:
                 target_col = cols[0]
             else:
                 return {"error": f"Target column for {ticker} not found."}

        close_series = features_df[target_col]
        y_logret_ahead = np.log(close_series).diff().shift(-1).rename("y_log_return_t+1")
        
        # Filter for target
        y = y_logret_ahead
        
        # Join and DropNA
        data = features_df.join(y).dropna()
        X = data.drop(columns=[y.name])
        y = data[y.name]
        
        if X.empty:
            return {"error": "Not enough data to train."}

        # 4. Feature Selection
        # Use all available features for this specific asset
        selected_features = X.columns.tolist()
        
        # 5. Train Model
        print(f"Training model on {len(X)} samples...")
        trainer = ModelTrainer()
        # We don't need to save the model artifact for this dynamic run
        trainer.train(X, y, selected_features, task="regression", save_model=False)
        
        # 6. Predict Next Day
        latest_features = features_df.iloc[[-1]][selected_features]
        
        if latest_features.isnull().values.any():
             latest_features = latest_features.fillna(method='ffill')
        
        pred_log_return = trainer.predict_next_day(latest_features)
        
        # Convert Log Return to Percentage
        pred_pct = (np.exp(pred_log_return) - 1) * 100
        
        # Calculate Price Levels
        current_price = close_series.iloc[-1]
        
        # Get USD/IDR
        try:
            usd_idr = yf.Ticker("IDR=X").history(period="1d")['Close'].iloc[-1]
        except:
            usd_idr = 15000.0
            
        current_price_idr = current_price * usd_idr
        
        # Calculate Volatility (Standard Deviation of Log Returns)
        # This gives us a baseline for what constitutes a "significant" move for this specific asset
        volatility = y.std()
        if np.isnan(volatility) or volatility == 0:
            volatility = 0.02 # Default to 2% if calculation fails
            
        # Dynamic Thresholds
        # We require the predicted move to be at least 0.5 standard deviations to be considered a signal
        # This filters out noise
        threshold = 0.5 * volatility
        
        print(f"Asset Volatility (std): {volatility:.4f}, Threshold: {threshold:.4f}")
        print(f"Predicted Return: {pred_log_return:.4f}")

        # ATR for SL/TP
        atr_col = f"{ticker}_atr14"
        if atr_col in features_df.columns:
            atr = features_df[atr_col].iloc[-1]
        else:
            atr = current_price * 0.05
            
        # Strategy Logic with Dynamic Thresholds
        if pred_log_return > threshold:
            direction = "BULLISH"
            sl = current_price - (1.5 * atr)
            tp = current_price + (2.0 * atr)
        elif pred_log_return < -threshold:
            direction = "BEARISH"
            sl = current_price + (1.5 * atr)
            tp = current_price - (2.0 * atr)
        else:
            direction = "NEUTRAL"
            sl = 0.0
            tp = 0.0
            
        # Calculate Predicted Price
        predicted_price_usd = current_price * np.exp(pred_log_return)
        predicted_price_idr = predicted_price_usd * usd_idr

        return {
            "ticker": ticker,
            "direction": direction,
            "pred_return": pred_log_return,
            "pred_pct": pred_pct,
            "volatility": volatility,
            "threshold": threshold,
            "current_price_usd": current_price,
            "current_price_idr": current_price_idr,
            "predicted_price_usd": predicted_price_usd,
            "predicted_price_idr": predicted_price_idr,
            "sl_idr": sl * usd_idr,
            "tp_idr": tp * usd_idr,
            "sl_usd": sl,
            "tp_usd": tp
        }

    except Exception as e:
        return {"error": f"Training/Prediction failed: {e}"}

if __name__ == "__main__":
    # Test
    print(train_and_predict("BTC-USD"))
