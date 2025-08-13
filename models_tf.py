from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------- windowing ----------

def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 60,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn (T, F) arrays into samples of shape (N, lookback, F) with next-horizon targets.
    """
    T = X.shape[0]
    Xs, ys = [], []
    for t in range(lookback, T - horizon + 1):
        Xs.append(X[t - lookback:t])
        ys.append(y[t + horizon - 1])
    return np.asarray(Xs), np.asarray(ys)


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------- LSTM model ----------

def build_lstm(input_dim: int,
               lookback: int,
               units: int = 128,
               dropout: float = 0.1) -> Model:
    """
    Simple stacked LSTM for regression.
    """
    inp = layers.Input(shape=(lookback, input_dim))
    x = layers.LSTM(units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units // 2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 2, activation="relu")(x)
    out = layers.Dense(1)(x)
    return Model(inp, out, name="lstm_regressor")


# ---------- Transformer model ----------

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.mha(x, x, attention_mask=None, training=training)
        x = self.ln1(x + self.do1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + self.do2(ffn_out, training=training))
        return x

def build_transformer(input_dim: int,
                      lookback: int,
                      d_model: int = 64,
                      num_heads: int = 4,
                      d_ff: int = 128,
                      num_layers: int = 2,
                      dropout: float = 0.1) -> Model:
    """
    Lightweight Transformer encoder for sequence regression.
    """
    inp = layers.Input(shape=(lookback, input_dim))
    # simple linear "embedding"
    x = layers.Dense(d_model)(inp)
    # add learnable positional encoding
    pos_emb = tf.Variable(tf.random.normal([1, lookback, d_model]), trainable=True, name="pos_emb")
    x = x + pos_emb
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, d_ff, dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_ff, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    return Model(inp, out, name="transformer_regressor")


# ---------- train / evaluate ----------

def compile_and_train(model: Model,
                      train_ds: tf.data.Dataset,
                      val_ds: Optional[tf.data.Dataset] = None,
                      lr: float = 1e-3,
                      epochs: int = 20,
                      loss: str = "mse") -> Model:
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["mae"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    return model

def predict(model: Model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return model.predict(X, batch_size=batch_size).squeeze()


# ---------- example wiring ----------

if __name__ == "__main__":
    # Fake data: (T, F)
    T, F = 2000, 12
    lookback, horizon = 60, 1
    rng = np.random.default_rng(42)
    Xraw = rng.normal(size=(T, F)).astype(np.float32)
    # Target: noisy linear combo of features + slight delay
    yraw = (Xraw[:, :3].sum(axis=1) + 0.1 * rng.normal(size=T)).astype(np.float32)

    Xw, yw = make_windows(Xraw, yraw, lookback=lookback, horizon=horizon)
    n = len(Xw)
    n_train, n_val = int(n*0.7), int(n*0.85)
    train_ds = make_tf_dataset(Xw[:n_train], yw[:n_train], batch_size=64, shuffle=True)
    val_ds   = make_tf_dataset(Xw[n_train:n_val], yw[n_train:n_val], batch_size=64, shuffle=False)
    test_X, test_y = Xw[n_val:], yw[n_val:]

    # LSTM
    lstm = build_lstm(input_dim=F, lookback=lookback, units=128, dropout=0.1)
    lstm = compile_and_train(lstm, train_ds, val_ds, lr=1e-3, epochs=10)
    preds_lstm = predict(lstm, test_X)
    print("LSTM test MAE:", np.mean(np.abs(preds_lstm - test_y)))

    # Transformer
    tr = build_transformer(input_dim=F, lookback=lookback, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout=0.1)
    tr = compile_and_train(tr, train_ds, val_ds, lr=1e-3, epochs=10)
    preds_tr = predict(tr, test_X)
    print("Transformer test MAE:", np.mean(np.abs(preds_tr - test_y)))







Ask ChatGPT
