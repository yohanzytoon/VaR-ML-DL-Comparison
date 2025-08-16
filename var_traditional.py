from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .utils import empirical_quantile

def var_parametric(returns: pd.Series, test_index: pd.DatetimeIndex, window: int, alpha: float) -> pd.Series:
    z = norm.ppf(alpha)
    out = pd.Series(index=test_index, dtype=float)
    for t in test_index:
        hist = returns[:t].iloc[-window:]
        mu = hist.mean()
        sigma = hist.std(ddof=1)
        out.loc[t] = -(mu + z * sigma)
    return out

def var_historical(returns: pd.Series, test_index: pd.DatetimeIndex, window: int, alpha: float) -> pd.Series:
    out = pd.Series(index=test_index, dtype=float)
    for t in test_index:
        hist = returns[:t].iloc[-window:]
        q = empirical_quantile(hist.values, alpha)
        out.loc[t] = -q
    return out

def var_monte_carlo(returns: pd.Series, test_index: pd.DatetimeIndex, window: int, alpha: float, n_sims: int = 10000) -> pd.Series:
    out = pd.Series(index=test_index, dtype=float)
    rng = np.random.default_rng(42)
    for t in test_index:
        hist = returns[:t].iloc[-window:]
        mu, sigma = hist.mean(), hist.std(ddof=1)
        sim = rng.standard_normal(n_sims) * sigma + mu
        q = np.quantile(sim, alpha)
        out.loc[t] = -q
    return out
