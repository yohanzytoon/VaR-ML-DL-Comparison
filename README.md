# VaR Project: Traditional vs ML vs DL

End-to-end, **reproducible** pipeline to compute and backtest 1% daily VaR using:
- Traditional methods: Parametric (variance–covariance), Historical, Monte Carlo
- ML: LightGBM for volatility (ML-Parametric) and direct quantile (ML-Quantile)
- DL: LSTM/GRU for volatility (DL-Parametric) and LSTM with pinball loss for direct quantile (DL-Quantile)
- AE-Adjust: Autoencoder anomaly-based VaR widening

## Quick start

```bash
# 1) (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Edit config
#    - By default, we use synthetic data so you can run without internet.
#    - To use real data, set data.use_synthetic: false (requires internet for yfinance).

# 4) Run a full backtest
python -m src.cli backtest --config config.yaml

# 5) Results
#    - Outputs: ./outputs/backtests/summary.csv and plots under ./outputs/plots
```

## Structure

```
var_project/
  ├─ src/
  │   ├─ data.py               # data loading (synthetic or yfinance)
  │   ├─ features.py           # feature engineering for ML/DL
  │   ├─ var_traditional.py    # parametric, historical, Monte Carlo VaR
  │   ├─ ml_models.py          # LightGBM pipelines for ML-Param, ML-Quantile
  │   ├─ dl_models.py          # Keras LSTM models for DL-Param, DL-Quantile
  │   ├─ ae_adjust.py          # autoencoder anomaly adjuster
  │   ├─ backtests.py          # Kupiec, Christoffersen, evaluation
  │   ├─ utils.py              # shared helpers
  │   └─ cli.py                # command-line entry point
  ├─ data/
  │   └─ synthetic_returns.csv # generated synthetic returns for quick run
  ├─ reports/
  │   └─ var_report_template.tex
  ├─ scripts/
  │   └─ run_backtest.sh
  ├─ config.yaml
  ├─ requirements.txt
  └─ README.md
```

## Notes

- **Internet** is required only if you set `data.use_synthetic: false` (for yfinance).
- Deep learning training can be slow on CPU; start with fewer epochs for a quick run.
- For publication-quality analysis, export outputs and compile `reports/var_report_template.tex`.
