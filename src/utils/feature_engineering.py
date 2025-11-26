import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def compute_log_returns(df_multi):
    """Compute log returns for multi-index DataFrame."""
    rets = {}
    for tkr in df_multi.columns.get_level_values(1).unique():
        # Check if 'Close' exists for this ticker
        if (tkr, 'Close') in df_multi.columns:
            close = df_multi[tkr, 'Close'].dropna()
        elif ('Close', tkr) in df_multi.columns:
             close = df_multi['Close', tkr].dropna()
        else:
            continue
            
        rets[tkr] = np.log(close).diff()
    out = pd.DataFrame(rets)
    out.index.name = "Date"
    return out

def build_features_from_price(df_daily, df_weekly, df_intra=None, assets=None):
    """Build features from daily, weekly, and intraday price data."""
    if assets is None:
        # Try to infer assets from columns
        if df_daily.columns.nlevels > 1:
             assets = df_daily.columns.get_level_values(0).unique()
        else:
             return pd.DataFrame()

    features = {}
    
    for tkr in assets:
        try:
            # Use xs to extract data for the specific ticker, assuming level 0 is Ticker
            # We check if Ticker is in level 0
            if tkr in df_daily.columns.get_level_values(0):
                df_tkr = df_daily.xs(tkr, level=0, axis=1)
            elif tkr in df_daily.columns.get_level_values(1):
                df_tkr = df_daily.xs(tkr, level=1, axis=1)
            else:
                # Ticker not found
                continue
                
            close = df_tkr['Close'].rename(f"{tkr}_close")
            high = df_tkr['High'].rename(f"{tkr}_high")
            low = df_tkr['Low'].rename(f"{tkr}_low")
            vol = df_tkr['Volume'].rename(f"{tkr}_volume")

        except KeyError:
            continue

        # lagged returns
        logret = np.log(close).diff().rename(f"{tkr}_logret")
        feats = pd.concat([close, high, low, vol, logret], axis=1)

        # rolling stats
        feats[f"{tkr}_roll_mean_7"] = close.rolling(7).mean()
        feats[f"{tkr}_roll_std_7"] = logret.rolling(7).std()
        feats[f"{tkr}_roll_mean_21"] = close.rolling(21).mean()
        feats[f"{tkr}_roll_std_21"] = logret.rolling(21).std()
        feats[f"{tkr}_autocorr_1"] = logret.rolling(30).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )

        # technical indicators
        rsi = RSIIndicator(close=close, window=14).rsi().rename(f"{tkr}_rsi14")
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        feats[f"{tkr}_macd"] = macd.macd()
        feats[f"{tkr}_macd_signal"] = macd.macd_signal()
        feats[f"{tkr}_macd_hist"] = macd.macd_diff()
        sma20 = SMAIndicator(close=close, window=20).sma_indicator().rename(f"{tkr}_sma20")
        ema20 = EMAIndicator(close=close, window=20).ema_indicator().rename(f"{tkr}_ema20")
        bb = BollingerBands(close=close, window=20, window_dev=2)
        feats[f"{tkr}_bb_high"] = bb.bollinger_hband()
        feats[f"{tkr}_bb_low"] = bb.bollinger_lband()
        atr = AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range().rename(f"{tkr}_atr14")

        feats = pd.concat([feats, rsi, sma20, ema20, atr], axis=1)

        # weekly trend features
        if df_weekly is not None and (tkr, 'Close') in df_weekly.columns:
            w_close = df_weekly[tkr, 'Close'].rename(f"{tkr}_w_close")
            feats = feats.join(w_close.reindex(feats.index).ffill())

        # intraday microstructure
        if df_intra is not None and (tkr, 'Close') in df_intra.columns:
            intra_close = df_intra[tkr, 'Close'].dropna()
            i_rets = np.log(intra_close).diff()
            i_df = (
                i_rets.groupby(i_rets.index.date)
                .std()
                .rename(f"{tkr}_i_vol_std")
                .to_frame()
            )
            i_df.index = pd.to_datetime(i_df.index)
            feats = feats.join(i_df)

        features[tkr] = feats

    all_feats = pd.concat(features.values(), axis=1)
    return all_feats
