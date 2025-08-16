from __future__ import annotations
import argparse, yaml
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm

from .data import load_returns
from .features import build_ml_features
from .var_traditional import var_parametric, var_historical, var_monte_carlo
from .ml_models import train_ml_parametric, predict_var_ml_parametric, train_ml_quantile, predict_var_ml_quantile
from .backtests import summarize_backtests
from .utils import ensure_dirs

def run_backtest(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = cfg["outputs"]["root"]
    out_plots = cfg["outputs"]["plots"]
    out_bk = cfg["outputs"]["backtests"]
    ensure_dirs(out_root, out_plots, out_bk)

    # Data
    use_syn = cfg["data"]["use_synthetic"]
    tickers = cfg["data"]["tickers"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    assets, port = load_returns(use_syn, tickers, start_date, end_date, data_dir="./data")

    # Splits
    train_end = cfg["splits"]["train_end"]
    val_end   = cfg["splits"]["val_end"]
    test_start= cfg["splits"]["test_start"]
    rets_train = port[:train_end]
    rets_val   = port[train_end:val_end]
    rets_test  = port[test_start:]

    # Traditional VaR
    alpha = cfg["var"]["alpha"]
    window = cfg["var"]["window"]
    n_sims = cfg["var"]["monte_carlo_sims"]
    var_param = var_parametric(port, rets_test.index, window, alpha)
    var_hist  = var_historical(port, rets_test.index, window, alpha)
    var_mc    = var_monte_carlo(port, rets_test.index, window, alpha, n_sims)

    # Features
    feats = build_ml_features(port, vix=None, window_list=cfg["ml"]["feature_windows"], L=cfg["ml"]["L_lags"])

    # ML Parametric
    z = norm.ppf(alpha)
    mlp_model = train_ml_parametric(feats, port, train_end, val_end, K=cfg["ml"]["vol_target_window"])
    var_mlp = pd.Series(index=rets_test.index, dtype=float)
    for t in rets_test.index:
        mu_t = port[:t].iloc[-window:].mean()
        x_t = feats.loc[t]
        var_mlp.loc[t] = predict_var_ml_parametric(mlp_model, x_t, mu_t, z)

    # ML Quantile
    mlq_model = train_ml_quantile(feats, port, train_end, val_end, alpha=alpha)
    var_mlq = pd.Series(index=rets_test.index, dtype=float)
    for t in rets_test.index:
        x_t = feats.loc[t]
        var_mlq.loc[t] = predict_var_ml_quantile(mlq_model, x_t)

    # Collect and summarize
    var_dict = {
        "Parametric": var_param,
        "Historical": var_hist,
        "MonteCarlo": var_mc,
        "MLParam": var_mlp,
        "MLQuantile": var_mlq
    }
    summary = summarize_backtests(rets_test, var_dict, alpha)
    out_csv = Path(out_bk) / "summary.csv"
    summary.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    return summary

def main():
    ap = argparse.ArgumentParser(description="VaR Project CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="Run full backtest (traditional + ML)")
    b.add_argument("--config", required=True, help="Path to config.yaml")

    args = ap.parse_args()
    if args.cmd == "backtest":
        run_backtest(args.config)

if __name__ == "__main__":
    main()
