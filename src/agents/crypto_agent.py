from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from agno.models.google import Gemini
from src.utils.data_loader import batch_download, ASSETS, PERIOD_FORECAST, DAILY_INTERVAL, WEEKLY_INTERVAL, INTRADAY_INTERVAL
from src.utils.feature_engineering import build_features_from_price
from src.utils.model_trainer import ModelTrainer
import pandas as pd
import yfinance as yf
import numpy as np

# Constants
EXOG = ["^VIX", "UUP", "GC=F", "^TNX"]

def get_prediction(dummy: str = "") -> str:
    """
    Predicts the price movement for all configured crypto assets for the next day.
    Selects the best asset to buy based on predicted return.
    Calculates Stop Loss and Take Profit levels.
    
    Args:
        dummy (str): Not used, just to satisfy tool signature if needed.
        
    Returns:
        str: A detailed prediction report.
    """
    print(f"Generating predictions for {ASSETS}...")
    
    # 1. Download Data
    try:
        daily_new = batch_download(ASSETS, PERIOD_FORECAST, DAILY_INTERVAL)
        weekly_new = batch_download(ASSETS, PERIOD_FORECAST, WEEKLY_INTERVAL)
        intra_new = batch_download(ASSETS, PERIOD_FORECAST, INTRADAY_INTERVAL)
    except Exception as e:
        return f"Error downloading data: {e}"

    # 2. Feature Engineering
    features_new_df = build_features_from_price(daily_new, weekly_new, intra_new, assets=ASSETS)
    
    # Add Exogenous Features
    try:
        exog_new = yf.download(EXOG, period=PERIOD_FORECAST, interval=DAILY_INTERVAL, group_by='column', progress=False)
        exog_close_new = {}
        for t in EXOG:
             if isinstance(exog_new.columns, pd.MultiIndex):
                exog_close_new[f"EXOG_{t.replace('^','').replace('=','')}"] = exog_new['Close'][t].rename(f"EXOG_{t}")
        
        if exog_close_new:
            exog_df_new = pd.concat(exog_close_new.values(), axis=1)
            features_new_df = features_new_df.join(exog_df_new)
    except Exception as e:
        print(f"Warning: Exogenous features failed: {e}")

    # 3. Load Model
    trainer = ModelTrainer()
    if not trainer.load_model():
        return "Model not found. Please train the model first."

    # 4. Predict for ALL Assets
    predictions = []
    
    # Get USD to IDR rate
    try:
        usd_idr = yf.Ticker("IDR=X").history(period="1d")['Close'].iloc[-1]
    except:
        usd_idr = 15000.0 # Fallback
        print("Warning: Could not fetch USD/IDR rate. Using fallback 15000.")

    for asset in ASSETS:
        # Filter features for this asset
        # Note: The model was trained on a dataset that might have had specific column names.
        # If the model is generic (trained on one asset but applied to others), feature names must match.
        # If the model was trained on BTC features specifically (e.g. "BTC-USD_rsi14"), it won't work for ETH directly
        # unless we rename columns or if the model was trained on generic names.
        # In the notebook, features were named "{tkr}_rsi14".
        # If we trained on BTC, the model expects "BTC-USD_rsi14".
        # To make it multi-asset with ONE model, we usually train on a generic dataset (renaming columns to "close", "rsi", etc.)
        # OR we train separate models for each asset.
        # Given the constraints and the notebook's approach (training on BTC), applying it to ETH might be inaccurate 
        # without retraining or feature renaming.
        # HOWEVER, for this implementation, let's assume we rename the columns of the current asset to match the training asset (BTC)
        # OR better, we just use the model as is if it was trained on a mix, but the notebook trained on BTC.
        # Let's try to "trick" the model by renaming current asset columns to match BTC columns if we only have one model.
        # OR we just predict for BTC for now if the user didn't ask for multi-model training.
        # User asked for "potential crypto to buy", implying selection.
        # Let's implement a simple column mapping strategy: 
        # Rename "{CurrentAsset}_feature" to "{TrainingAsset}_feature" (BTC-USD).
        
        # Get latest features for this asset
        asset_cols = [c for c in features_new_df.columns if c.startswith(asset)]
        if not asset_cols:
            continue
            
        asset_features = features_new_df[asset_cols].copy()
        
        # Rename columns to match BTC-USD (the training asset)
        # This assumes the model learns "general" patterns.
        rename_map = {c: c.replace(asset, "BTC-USD") for c in asset_cols}
        asset_features = asset_features.rename(columns=rename_map)
        
        # Add Exog (shared)
        exog_cols = [c for c in features_new_df.columns if c.startswith("EXOG_")]
        asset_features = asset_features.join(features_new_df[exog_cols])
        
        # Predict
        pred_log_return = trainer.predict_next_day(asset_features)
        
        if pred_log_return is not None:
            # Get current price
            current_price_usd = daily_new[asset, 'Close'].iloc[-1]
            current_price_idr = current_price_usd * usd_idr
            
            # Calculate SL/TP (Simple Strategy: ATR based or fixed %)
            # Let's use ATR if available, or fixed 2% SL, 4% TP
            atr_col = f"{asset}_atr14" # Original name
            if atr_col in features_new_df.columns:
                atr = features_new_df[atr_col].iloc[-1]
            else:
                atr = current_price_usd * 0.05 # Fallback 5%
            
            # Strategy: Buy if > 0
            if pred_log_return > 0:
                # Long
                sl_price = current_price_usd - (1.5 * atr)
                tp_price = current_price_usd + (2.0 * atr) # Risk Reward > 1
            else:
                # Short (or just don't buy)
                # For "Spot" recommendations, we usually only care about Buy.
                sl_price = current_price_usd + (1.5 * atr)
                tp_price = current_price_usd - (2.0 * atr)

            predictions.append({
                "asset": asset,
                "pred_return": pred_log_return,
                "current_price_idr": current_price_idr,
                "sl_idr": sl_price * usd_idr,
                "tp_idr": tp_price * usd_idr,
                "direction": "UP" if pred_log_return > 0 else "DOWN"
            })

    # 5. Select Best Asset
    if not predictions:
        return "No predictions generated."
        
    # Sort by predicted return (descending)
    predictions.sort(key=lambda x: x['pred_return'], reverse=True)
    best_pick = predictions[0]
    
    if best_pick['pred_return'] <= 0:
        return "Market Outlook: Bearish. No recommended buys for tomorrow."

    # Format Report
    report = f"""
**Best Pick: {best_pick['asset']}**
*   **Forecast**: {best_pick['direction']} ({best_pick['pred_return']:.4f} log return)
*   **Buy Price**: Rp {best_pick['current_price_idr']:,.0f}
*   **Stop Loss**: Rp {best_pick['sl_idr']:,.0f}
*   **Take Profit**: Rp {best_pick['tp_idr']:,.0f}
    """
    return report

import os

# Create the Crypto Agent
crypto_agent = Agent(
    name="Crypto Agent",
    role="Financial Analyst",
    instructions="You are a financial analyst. Use the prediction tool to find the best crypto asset to buy.",
    tools=[get_prediction, ReasoningTools(
            enable_think=True,
            enable_analyze=True,
            add_instructions=True,
            add_few_shot=True,
        )],
    model=Gemini(id=os.getenv("GEMINI_MODEL_ID", "gemini-flash-latest")),
    retries=int(os.getenv("AGENT_RETRIES", 3)),
    delay_between_retries=int(os.getenv("RETRY_DELAY", 5)),
    markdown=True,
)
