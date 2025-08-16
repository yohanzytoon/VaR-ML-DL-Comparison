from __future__ import annotations
import pandas as pd
import numpy as np

def build_ml_features(returns: pd.Series, vix: pd.Series = None,
                      window_list=[5,10,20], L=5) -> pd.DataFrame:
    df = pd.DataFrame(index=returns.index)
    for w in window_list:
        df[f"vol_{w}"] = returns.rolling(window=w).std()

    for lag in range(1, L+1):
        df[f"abs_r_{lag}"] = returns.shift(lag).abs()
        df[f"sqr_r_{lag}"] = (returns.shift(lag))**2

    if vix is not None:
        df["VIX"] = vix.reindex(df.index).ffill()

    dow = returns.index.dayofweek
    for i in range(5):
        df[f"DOW_{i}"] = (dow == i).astype(int)

    month = returns.index.month
    for m in range(1, 13):
        df[f"Month_{m}"] = (month == m).astype(int)

    return df.dropna()
