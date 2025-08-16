import os
import pandas as pd
import numpy as np
from pathlib import Path

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def rolling_mean_std(series: pd.Series, window: int):
    mu = series.rolling(window=window).mean()
    sigma = series.rolling(window=window).std(ddof=1)
    return mu, sigma

def empirical_quantile(arr, q):
    # numpy's percentiles can be sensitive; use nanpercentile for safety
    return np.nanpercentile(np.asarray(arr), q * 100.0)
