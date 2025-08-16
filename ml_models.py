from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

def train_ml_parametric(features: pd.DataFrame, returns: pd.Series,
                        train_end: str, val_end: str, K: int = 5):
    # Target: realized vol over next K days (shift -1 to align with features)
    target = returns.rolling(K).std().shift(-1).rename("target_vol")
    df = pd.concat([features, target], axis=1).dropna()
    df_train = df[:train_end]
    df_val = df[train_end:val_end]

    X_train, y_train = df_train.drop(columns="target_vol"), df_train["target_vol"]
    X_val, y_val     = df_val.drop(columns="target_vol"), df_val["target_vol"]

    model = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric="l2")
    return model

def predict_var_ml_parametric(model, features_row: pd.Series, mu_t: float, z_alpha: float) -> float:
    sigma_hat = float(model.predict(features_row.values.reshape(1, -1)))
    return -(mu_t + z_alpha * sigma_hat)

def train_ml_quantile(features: pd.DataFrame, returns: pd.Series,
                      train_end: str, val_end: str, alpha: float = 0.01):
    # Target: next-day return
    target = returns.shift(-1).rename("target_ret")
    df = pd.concat([features, target], axis=1).dropna()
    df_train = df[:train_end]
    df_val = df[train_end:val_end]

    X_train, y_train = df_train.drop(columns="target_ret"), df_train["target_ret"]
    X_val, y_val     = df_val.drop(columns="target_ret"), df_val["target_ret"]

    model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2"
    )

    return model

def predict_var_ml_quantile(model, features_row: pd.Series) -> float:
    q_hat = float(model.predict(features_row.values.reshape(1, -1)))
    return -q_hat
