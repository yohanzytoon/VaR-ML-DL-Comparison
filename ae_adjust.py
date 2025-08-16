from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim: int, encoding_dim: int = 16):
    ae_input = Input(shape=(input_dim,), name="ae_input")
    e = Dense(units=32, activation="relu")(ae_input)
    bottleneck = Dense(units=encoding_dim, activation="relu", name="bottleneck")(e)
    d = Dense(units=32, activation="relu")(bottleneck)
    rec = Dense(units=input_dim, activation="linear", name="reconstruction")(d)
    model = Model(inputs=ae_input, outputs=rec)
    model.compile(optimizer="adam", loss="mse")
    return model

def anomaly_multiplier(ae_model, x_vec: np.ndarray, median_eps: float, mad_eps: float, c_thresh: float, lam: float) -> float:
    x_rec = ae_model.predict(x_vec.reshape(1,-1), verbose=False)
    eps_t = float(np.mean((x_vec - x_rec.reshape(-1))**2))
    eps_std = (eps_t - median_eps) / (mad_eps if mad_eps != 0 else 1.0)
    m_t = 1.0 + lam * max(0.0, eps_std - c_thresh)
    return m_t, eps_t
