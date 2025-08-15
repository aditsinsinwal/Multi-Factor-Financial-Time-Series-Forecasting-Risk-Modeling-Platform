# Train TF models on engineered features from your DB.

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from typing import Tuple, List
from sqlalchemy import create_engine
import tensorflow as tf
from models_tf import make_windows, make_tf_dataset, build_lstm, build_transformer, compile_and_train, predict

# --- config ---
DB_URL          = os.getenv("DB_URL", "sqlite:///market_data.db")
FEATURE_TABLE   = os.getenv("FEATURE_TABLE", "features")
TARGET_COL      = os.getenv("TARGET_COL", "ret_1d_t1")  # from to_supervised(..., horizon=1)
LOOKBACK        = int(os.getenv("LOOKBACK", 60))
MODEL_TYPE      = os.getenv("MODEL_TYPE", "lstm")       # "lstm" or "transformer"
EPOCHS          = int(os.getenv("EPOCHS", 20))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", 64))
OUT_DIR         = os.getenv("OUT_DIR", "models")
TRAIN_RATIO     = float(os.getenv("TRAIN_RATIO", 0.7))
VAL_RATIO       = float(os.getenv("VAL_RATIO", 0.15))
SEED            = int(os.getenv("SEED", 42))

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

# --- data utils ---

def load_features(db_url: str, table: str) -> pd.DataFrame:
    eng = create_engine(db_url, future=True)
    df = pd.read_sql_table(table, eng, parse_dates=["date"])
    df = df.sort_values(["symbol", "date"])
    # keep numeric features + id cols
    id_cols = ["date", "symbol"]
    num = df.select_dtypes(include=["number"]).columns.tolist()
    df = df[id_cols + num]
    return df

def split_idx(n: int, train_ratio: float, val_ratio: float) -> Tuple[slice, slice, slice]:
    n_train = int(n * train_ratio)
    n_val   = int(n * (train_ratio + val_ratio))
    return slice(0, n_train), slice(n_train, n_val), slice(n_val, n)

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = X.mean(axis=(0,1), keepdims=True)
    s = X.std(axis=(0,1), keepdims=True) + 1e-8
    return m, s, (X - m) / s

def standardize_apply(X: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (X - m) / s

# --- training one symbol ---

def train_one_symbol(df_sym: pd.DataFrame, feature_cols: List[str]) -> dict:
    y = df_sym[TARGET_COL].values.astype("float32")
    X = df_sym[feature_cols].values.astype("float32")
    # windowing
    Xw, yw = make_windows(X, y, lookback=LOOKBACK, horizon=1)
    n = len(Xw)
    if n < 200:  # not enough samples
        return {"status": "skipped", "reason": "too_few_samples", "n": n}
    tr, va, te = split_idx(n, TRAIN_RATIO, VAL_RATIO)
    Xtr, ytr = Xw[tr], yw[tr]
    Xva, yva = Xw[va], yw[va]
    Xte, yte = Xw[te], yw[te]
    # scale per symbol from train only
    mean, std, Xtr = standardize_fit(Xtr)
    Xva = standardize_apply(Xva, mean, std)
    Xte = standardize_apply(Xte, mean, std)

    # model
    F = Xtr.shape[-1]
    if MODEL_TYPE == "transformer":
        model = build_transformer(input_dim=F, lookback=LOOKBACK, d_model=64, num_heads=4, d_ff=128, num_layers=2, dropout=0.1)
    else:
        model = build_lstm(input_dim=F, lookback=LOOKBACK, units=128, dropout=0.1)

    # datasets
    ds_tr = make_tf_dataset(Xtr, ytr, batch_size=BATCH_SIZE, shuffle=True)
    ds_va = make_tf_dataset(Xva, yva, batch_size=BATCH_SIZE, shuffle=False)

    # train
    model = compile_and_train(model, ds_tr, ds_va, lr=1e-3, epochs=EPOCHS, loss="mse")

    # eval
    preds = model.predict(Xte, batch_size=256).squeeze()
    mae = float(np.mean(np.abs(preds - yte)))
    mse = float(np.mean((preds - yte) ** 2))

    # save artifacts
    sym = df_sym["symbol"].iloc[0]
    base = os.path.join(OUT_DIR, f"{sym}_{MODEL_TYPE}")
    model.save(base + ".keras")
    np.savez(base + "_scaler.npz", mean=mean.squeeze(), std=std.squeeze(), features=np.array(feature_cols))
    meta = {"symbol": sym, "model": MODEL_TYPE, "lookback": LOOKBACK, "features": feature_cols, "mae": mae, "mse": mse}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return {"status": "ok", **meta}

# --- main ---

if __name__ == "__main__":
    df = load_features(DB_URL, FEATURE_TABLE)
    # pick features = numeric cols excluding target and obvious leakage
    drop = {TARGET_COL, "px", "Volume"}  # adjust as needed
    num_cols = df.select_dtypes(include=["number"]).columns
    feat_cols = sorted([c for c in num_cols if c not in drop])

    results = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        # remove rows with any missing features after creation
        g = g.dropna(subset=feat_cols)
        if len(g) < 500:
            results.append({"symbol": sym, "status": "skipped", "reason": "too_few_rows", "n": len(g)})
            continue
        res = train_one_symbol(g, feat_cols)
        results.append(res)

    out_df = pd.DataFrame(results).sort_values(["status", "mae"], na_position="last")
    out_df.to_csv(os.path.join(OUT_DIR, "training_summary.csv"), index=False)
    print(out_df)
