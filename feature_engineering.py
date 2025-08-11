from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Sequence
from db_utils import read_prices, read_macro, join_prices_macro, write_df


# ---------- core indicators ----------

def pct_returns(prices: pd.Series, periods: Sequence[int] = (1, 5, 10)) -> pd.DataFrame:
    out = {}
    for p in periods:
        out[f"ret_{p}d"] = prices.pct_change(p)
    return pd.DataFrame(out, index=prices.index)

def rolling_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    r = prices.pct_change()
    return r.rolling(window).std() * np.sqrt(252)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_pct_b(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return (prices - lower) / (upper - lower)

def add_lags(df: pd.DataFrame, cols: Iterable[str], lags: Sequence[int] = (1, 2, 5, 10)) -> pd.DataFrame:
    for c in cols:
        for l in lags:
            df[f"{c}_lag{l}"] = df[c].shift(l)
    return df


# ---------- feature builder ----------

def build_features(
    db_url: str = "sqlite:///market_data.db",
    symbols: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    price_col: str = "Adj Close",
    add_macro: bool = True,
) -> pd.DataFrame:
    """
    Returns tidy features with one row per (date, symbol).
    """
    # prices: wide matrix (date x symbol)
    prices_wide = read_prices(db_url, symbols=symbols, start=start, end=end, price_col=price_col, wide=True)

    # macro (optional)
    macro = read_macro(db_url, start=start, end=end) if add_macro else None
    if macro is not None and not macro.empty:
        joined = join_prices_macro(prices_wide, macro)  # daily align + ffill macro
    else:
        joined = prices_wide.copy()

    # build per-symbol features then stack
    feats = []
    for sym in prices_wide.columns:
        px = joined[sym].rename("price")

        f = pct_returns(px, periods=(1, 5, 10))
        f["vol_20d"] = rolling_vol(px, 20)
        f["rsi_14"] = rsi(px, 14)
        f = f.join(macd(px), how="left")
        f["bb_pctb_20_2"] = bollinger_pct_b(px, 20, 2.0)
        f["ma_20"] = px.rolling(20).mean()
        f["ma_50"] = px.rolling(50).mean()
        f["px"] = px

        # add lags for a few predictive cols
        f = add_lags(f, cols=["ret_1d", "rsi_14", "macd", "vol_20d"], lags=(1, 2, 5, 10))

        # attach macro columns (same index)
        if macro is not None and not macro.empty:
            f = f.join(macro, how="left")

        f["symbol"] = sym
        feats.append(f)

    out = pd.concat(feats).reset_index().rename(columns={"index": "date"})
    # sort + drop insane rows
    out = out.sort_values(["symbol", "date"]).replace([np.inf, -np.inf], np.nan).dropna(how="all", subset=["px"])

    return out


# ---------- helpers ----------

def train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split. Keeps order.
    """
    df = df.sort_values(date_col)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]


def to_supervised(
    df: pd.DataFrame,
    target_col: str = "ret_1d",   # predict next-day return by default
    horizon: int = 1,             # how many steps ahead
    group_col: str = "symbol",
    date_col: str = "date",
    drop_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Shift target by horizon within each symbol.
    """
    if drop_cols is None:
        drop_cols = []
    df = df.copy()
    df = df.sort_values([group_col, date_col])
    df[f"{target_col}_t{horizon}"] = df.groupby(group_col)[target_col].shift(-horizon)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    df = df.dropna(subset=[f"{target_col}_t{horizon}"])
    return df


def save_features_to_db(features: pd.DataFrame, db_url: str = "sqlite:///market_data.db", table: str = "features"):
    write_df(features, table, db_url, if_exists="append")


# ---------- run as script ----------

if __name__ == "__main__":
    DB = "sqlite:///market_data.db"
    # build
    feats = build_features(DB, symbols=None, start="2016-01-01", price_col="Adj Close", add_macro=True)
    # create a supervised label (predict 1d return)
    sup = to_supervised(feats, target_col="ret_1d", horizon=1, drop_cols=["px"])
    # optional: save
    save_features_to_db(sup, DB, table="features")
    print("Features shape:", sup.shape)
    print(sup.head())
