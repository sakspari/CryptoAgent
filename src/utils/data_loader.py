import time
import pandas as pd
import yfinance as yf
from pathlib import Path

# Constants
ASSETS = ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD"]
PERIOD_DAILY = "1y"
PERIOD_WEEKLY = "1y"
PERIOD_INTRADAY = "60d"
DAILY_INTERVAL = "1d"
WEEKLY_INTERVAL = "1wk"
INTRADAY_INTERVAL = "1h"
PERIOD_FORECAST = "90d"

def ensure_dirs():
    """Ensure data directories exist."""
    for interval in [DAILY_INTERVAL, WEEKLY_INTERVAL, INTRADAY_INTERVAL]:
        Path(f"data/raw/{interval}").mkdir(parents=True, exist_ok=True)

def normalize_columns_to_field_ticker(df):
    """Normalize DataFrame columns to (Ticker, Field) format."""
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        lvl0 = df.columns.get_level_values(0)
        # If level 0 contains Tickers, we are good.
        if any(x in ASSETS for x in lvl0):
            pass # Already (Ticker, Field)
        # If level 1 contains Tickers, swap.
        elif any(x in ASSETS for x in df.columns.get_level_values(1)):
            df = df.swaplevel(axis=1)
        df = df.sort_index(axis=1)
        return df
    return pd.concat({ "UNKNOWN": df }, axis=1).swaplevel(axis=1).sort_index(axis=1)

def safe_download_one(ticker:str, period:str, interval:str, sleep_sec:float=1.2, retries:int=4, backoff:float=1.6):
    """Download data for a single ticker with retries."""
    last_exc = None
    for i in range(retries):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="column",
                threads=False,
                progress=False,
            )
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            
            # Ensure directory exists before saving
            ensure_dirs()
            
            out_path = Path(f"data/raw/{interval}/{ticker}_{period}_{interval}.csv")
            df.to_csv(out_path)
            
            # Handle yfinance return structure
            # If yfinance returns MultiIndex (e.g. Price, Ticker), we want to standardize to (Ticker, Field)
            if isinstance(df.columns, pd.MultiIndex):
                # Check if we have (Price, Ticker)
                if df.columns.nlevels == 2:
                    # If level 1 is ticker, swap to (Ticker, Price)
                    # We assume level 0 is Price (Close, Open, etc)
                    # But we need to be careful.
                    # Let's just flatten and rebuild to be safe.
                    # Or check if ticker is in level 1
                    if ticker in df.columns.get_level_values(1):
                         df = df.swaplevel(axis=1)
                    # If ticker is in level 0, it is (Ticker, Price), good.
                else:
                    # > 2 levels? Drop extra levels?
                    # For now, let's try to just grab the Price level and wrap
                    pass
            else:
                # Flat columns (Open, Close...), wrap with Ticker
                df = pd.concat({ticker: df}, axis=1)

            # Final check: Ensure we have (Ticker, Field)
            # If we still have (Price, Ticker), swap
            if df.columns.nlevels == 2:
                 # If level 0 is NOT ticker, swap
                 if ticker not in df.columns.get_level_values(0):
                      df = df.swaplevel(axis=1)
            
            df = df.sort_index(axis=1)
            time.sleep(sleep_sec)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(sleep_sec * (backoff ** i))
    raise RuntimeError(f"failed to download {ticker} {period} {interval}: {last_exc}")

def batch_download(tickers, period, interval):
    """Download data for multiple tickers."""
    frames = []
    for t in tickers:
        frames.append(safe_download_one(t, period, interval))
    out = pd.concat(frames, axis=1)
    out = normalize_columns_to_field_ticker(out)
    return out
