#baselines: ARIMA, VAR, (optional) GARCH.
#using on (log) returns, not raw prices.

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

# GARCH is from the separate 'arch' package (pip install arch)
try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False
    warnings.warn("GARCH disabled: install 'arch' to enable.")


# ---------- helpers ----------

def to_returns(px: pd.Series, log: bool = True) -> pd.Series:
    r = np.log(px).diff() if log else px.pct_change()
    return r.dropna()

def align_index(*series: pd.Series) -> List[pd.Series]:
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return [s.reindex(idx).dropna() for s in series]


# ---------- ARIMA (univariate) ----------

def arima_fit(series: pd.Series, order: Tuple[int,int,int] = (1,0,1)) -> Any:
    model = ARIMA(series, order=order)
    return model.fit()

def arima_forecast(series: pd.Series,
                   order: Tuple[int,int,int] = (1,0,1),
                   steps: int = 1) -> Tuple[pd.Series, Any]:
    fit = arima_fit(series, order)
    fc = fit.forecast(steps=steps)
    fc.index = pd.RangeIndex(len(series), len(series) + steps)
    return fc, fit

def arima_walk_forward(series: pd.Series,
                       order: Tuple[int,int,int] = (1,0,1),
                       window: int = 252,
                       horizon: int = 1) -> pd.Series:
    """Expanding walk-forward forecast on returns."""
    series = series.dropna()
    preds = []
    idxs = []
    for t in range(window, len(series) - horizon + 1):
        train = series.iloc[:t]
        model = ARIMA(train, order=order).fit()
        fc = model.forecast(steps=horizon).iloc[-1]
        preds.append(fc)
        idxs.append(series.index[t + horizon - 1])
    return pd.Series(preds, index=idxs, name="arima_pred")


# ---------- VAR (multivariate) ----------

def var_fit(df_ret: pd.DataFrame, lags: int = 1) -> Any:
    model = VAR(df_ret.dropna())
    return model.fit(lags)

def var_forecast(df_ret: pd.DataFrame, lags: int = 1, steps: int = 1) -> Tuple[pd.DataFrame, Any]:
    fit = var_fit(df_ret, lags)
    fc = fit.forecast(fit.y, steps=steps)
    fc = pd.DataFrame(fc, columns=df_ret.columns)
    return fc, fit

def var_walk_forward(df_ret: pd.DataFrame,
                     lags: int = 1,
                     window: int = 252,
                     horizon: int = 1) -> pd.DataFrame:
    df_ret = df_ret.dropna()
    preds = []
    idxs = []
    for t in range(window, len(df_ret) - horizon + 1):
        train = df_ret.iloc[:t]
        fit = VAR(train).fit(lags)
        fc = fit.forecast(train.values[-lags:], steps=horizon)[-1]
        preds.append(fc)
        idxs.append(df_ret.index[t + horizon - 1])
    out = pd.DataFrame(preds, index=idxs, columns=df_ret.columns)
    out.columns = [f"var_pred_{c}" for c in out.columns]
    return out


# ---------- GARCH (volatility) ----------

def garch_fit(ret: pd.Series, p: int = 1, q: int = 1) -> Any:
    if not _HAS_ARCH:
        raise RuntimeError("Install 'arch' for GARCH.")
    am = arch_model(ret.dropna(), mean="constant", vol="GARCH", p=p, q=q, dist="normal")
    return am.fit(disp="off")

def garch_forecast(ret: pd.Series,
                   p: int = 1, q: int = 1,
                   steps: int = 1) -> Tuple[pd.Series, Any]:
    if not _HAS_ARCH:
        raise RuntimeError("Install 'arch' for GARCH.")
    fit = garch_fit(ret, p, q)
    fc = fit.forecast(horizon=steps, reindex=False)
    # variance forecasts (convert to vol)
    var = fc.variance.values[-1]
    vol = pd.Series(np.sqrt(var), name="garch_vol")
    return vol, fit


# ---------- convenience ----------

def baseline_summary(px: pd.Series,
                     order=(1,0,1),
                     garch_pq=(1,1),
                     window=252) -> Dict[str, Any]:
    """Quick baselines on a single asset."""
    ret = to_returns(px)
    arima_preds = arima_walk_forward(ret, order=order, window=window)
    out = {"returns": ret, "arima_pred": arima_preds}
    if _HAS_ARCH:
        vol_fc, _ = garch_forecast(ret, p=garch_pq[0], q=garch_pq[1], steps=5)
        out["garch_vol_5"] = vol_fc
    return out


if __name__ == "__main__":
    #test
    rng = pd.date_range("2022-01-01", periods=600, freq="B")
    px = pd.Series(100 + np.cumsum(np.random.randn(len(rng))*0.5), index=rng, name="PX")
    ret = to_returns(px)
    print(arima_walk_forward(ret).tail())
