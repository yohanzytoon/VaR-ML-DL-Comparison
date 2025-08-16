from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def load_returns(use_synthetic: bool,
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.Series]:
    """Return (asset_returns_df, equal_weight_portfolio_returns).
    If use_synthetic=True, loads ./data/synthetic_returns.csv
    Otherwise, downloads Adjusted Close via yfinance and converts to log-returns.
    """
    if use_synthetic:
        path = Path(data_dir) / "synthetic_returns.csv"
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        # Already returns
        asset_returns = df[tickers].copy()
    else:
        import yfinance as yf
        price = yf.download(tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
        price = price.dropna()
        price = price[tickers] if isinstance(price, pd.DataFrame) else price.to_frame(name=tickers[0])
        asset_returns = np.log(price / price.shift(1)).dropna()

    weights = np.array([1.0 / len(tickers)] * len(tickers))
    port_rets = asset_returns.dot(weights)
    port_rets.name = "portfolio_return"
    return asset_returns, port_rets
