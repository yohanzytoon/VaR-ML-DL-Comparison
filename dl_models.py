from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Concatenate

def build_dl_param_model(seq_len: int, f_aux: int, dropout: float = 0.2, lstm_units: int = 32, aux_units: int = 16):
    seq_input = Input(shape=(seq_len, 1), name="seq_input")
    aux_input = Input(shape=(f_aux,), name="aux_input")
    x = LSTM(units=lstm_units, return_sequences=False)(seq_input)
    x = Dropout(dropout)(x)
    y = Dense(units=aux_units, activation="relu")(aux_input)
    y = Dropout(dropout)(y)
    combined = Concatenate()([x, y])
    z = Dense(units=32, activation="relu")(combined)
    z = Dropout(dropout)(z)
    out = Dense(units=1, activation="relu", name="pred_vol")(z)
    model = Model(inputs=[seq_input, aux_input], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.01, **kwargs):
        super().__init__(**kwargs)
        self.q = tf.constant(quantile, dtype=tf.float32)
    def call(self, y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.q * e, (self.q - 1.0) * e))

def build_dl_quant_model(seq_len: int, f_aux: int, dropout: float = 0.2, lstm_units: int = 32, aux_units: int = 16, alpha: float = 0.01):
    seq_input = Input(shape=(seq_len, 1), name="seq_input_q")
    aux_input = Input(shape=(f_aux,), name="aux_input_q")
    x = LSTM(units=lstm_units, return_sequences=False)(seq_input)
    x = Dropout(dropout)(x)
    y = Dense(units=aux_units, activation="relu")(aux_input)
    y = Dropout(dropout)(y)
    combined = Concatenate()([x, y])
    z = Dense(units=32, activation="relu")(combined)
    z = Dropout(dropout)(z)
    out = Dense(units=1, activation="linear", name="pred_quantile")(z)
    model = Model(inputs=[seq_input, aux_input], outputs=out)
    model.compile(optimizer="adam", loss=QuantileLoss(alpha))
    return model
