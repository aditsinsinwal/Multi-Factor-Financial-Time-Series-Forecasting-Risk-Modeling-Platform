
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict

# --------- math utils ---------
def _safe_inv(a: np.ndarray, l2: float = 1e-8) -> np.ndarray:
    # ridge to avoid singularities
    n = a.shape[0]
    return np.linalg.inv(a + l2 * np.eye(n))

def _project_simplex(w: np.ndarray) -> np.ndarray:
    # project to {w_i >=0, sum w_i =1}
    if w.sum() <= 0:
        return np.ones_like(w) / len(w)
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones_like(w) / len(w)

def returns_from_prices(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    ret = np.log(prices).diff() if log else prices.pct_change()
    return ret.dropna(how="all")

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    # RiskMetrics EWMA covariance
    r = returns.values
    mu = np.zeros(r.shape[1])
    S = np.cov(r[:50].T)  # warm start
    for t in range(returns.shape[0]):
        x = r[t] - mu
        S = lam * S + (1 - lam) * np.outer(x, x)
    return S

# --------- optimizers ---------
def mean_variance_weights(mu: np.ndarray,
                          cov: np.ndarray,
                          risk_aversion: float = 1.0,
                          long_only: bool = True,
                          l2: float = 1e-6) -> np.ndarray:
    """
    Max mu^T w - 0.5*λ w^T Σ w - (l2/2)||w||^2. Closed form via linear solve.
    """
    A = risk_aversion * cov + l2 * np.eye(len(mu))
    w = _safe_inv(A) @ mu
    w = _project_simplex(w) if long_only else w / (w.sum() + 1e-12)
    return w

def minimum_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    inv = _safe_inv(cov)
    w = inv @ np.ones(len(cov))
    w /= w.sum()
    return _project_simplex(w) if long_only else w

def risk_parity_weights(cov: np.ndarray, iters: int = 10_000, tol: float = 1e-8) -> np.ndarray:
    # simple CCD on log-barrier objective
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        w_old = w.copy()
        for i in range(n):
            c = cov[i, i]
            b = cov[i, :] @ w - c * w[i]
            w[i] = np.sqrt(max(1e-12, (1 / c) * (w @ cov @ w) / n)) - b / c
            w[i] = max(w[i], 1e-12)
        w /= w.sum()
        if np.linalg.norm(w - w_old, 1) < tol:
            break
    return w

# --------- Black–Litterman (light) ---------
def black_litterman(mu_prior: np.ndarray,
                    cov: np.ndarray,
                    P: np.ndarray,
                    q: np.ndarray,
                    tau: float = 0.05,
                    omega: Optional[np.ndarray] = None) -> np.ndarray:
    """
    mu_post = inv(inv(tau*Σ) + P^T Ω^-1 P) (inv(tau*Σ) mu_prior + P^T Ω^-1 q)
    """
    n = len(mu_prior)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))  # diag uncertainty
    A = _safe_inv(tau * cov)
    B = P.T @ _safe_inv(omega) @ P
    rhs = A @ mu_prior + P.T @ _safe_inv(omega) @ q
    mu_post = _safe_inv(A + B) @ rhs
    return mu_post

# --------- turnover + scaling ---------
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w @ cov @ w))

def target_vol_scale(w: np.ndarray, cov: np.ndarray, target_vol: Optional[float]) -> np.ndarray:
    if target_vol is None:
        return w
    vol = portfolio_vol(w, cov)
    return w if vol < 1e-12 else w * (target_vol / vol)

def turnover(prev_w: np.ndarray, w: np.ndarray) -> float:
    return float(np.abs(w - prev_w).sum())

# --------- rebalancing loop ---------
def rebalance_portfolio(
    prices: pd.DataFrame,
    freq: str = "M",
    ret_kind: str = "simple",        # "simple" or "log"
    cov_lam: float = 0.94,
    weight_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = minimum_variance_weights,
    post_scale_target_vol: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Walk-forward: at each rebalance date, estimate mu (mean) and cov from past window, get weights, and hold until next.
    """
    px = prices.dropna(how="all").copy()
    rets = returns_from_prices(px, log=(ret_kind == "log"))
    rets = rets.loc[~rets.isna().all(axis=1)]
    dates = rets.resample(freq).last().index  # rebal points

    w_hist = []
    port_ret = pd.Series(index=rets.index, dtype=float)
    prev_w = np.zeros(px.shape[1])

    for i, d in enumerate(dates):
        hist = rets.loc[:d].dropna()
        if len(hist) < 60:  # not enough history
            continue
        mu = hist.mean().values  # simple mean; replace with your model forecasts if you have them
        cov = ewma_cov(hist, lam=cov_lam)

        w = weight_func(mu, cov) if weight_func.__code__.co_argcount == 2 else weight_func(cov)
        w = target_vol_scale(w, cov, post_scale_target_vol)

        # apply weights until next rebalance date
        if i < len(dates) - 1:
            seg = rets.loc[(rets.index > d) & (rets.index <= dates[i + 1])]
        else:
            seg = rets.loc[rets.index > d]
        pr = (seg.values @ w).astype(float)
        port_ret.loc[seg.index] = pr

        w_hist.append(pd.Series(w, index=px.columns, name=d))
        prev_w = w

    W = pd.DataFrame(w_hist).sort_index()
    return {
        "weights": W,
        "portfolio_returns": port_ret.dropna(),
        "turnover": W.diff().abs().sum(axis=1).fillna(0.0),
    }

# --------- demo ---------
if __name__ == "__main__":
    # toy price panel
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    n = 5
    shocks = rng.normal(size=(len(idx), n)) * 0.01
    px = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=idx, columns=[f"A{i}" for i in range(n)])

    res = rebalance_portfolio(px, freq="M", weight_func=minimum_variance_weights, post_scale_target_vol=0.12)
    print("Weights (head):")
    print(res["weights"].head())
    print("Turnover (mean):", res["turnover"].mean().round(4))
    print("Portfolio return mean:", res["portfolio_returns"].mean().round(6))
