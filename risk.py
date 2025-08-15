# Risk metrics + simple optimizer + simulation.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ---- basic metrics ----

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    rp = returns.mean() * periods_per_year
    sp = returns.std(ddof=1) * np.sqrt(periods_per_year)
    rf_a = rf
    return (rp - rf_a) / (sp + 1e-12)

def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    rf_p = rf / periods_per_year
    excess = returns - rf_p
    downside = excess[excess < 0]
    ds = downside.std(ddof=1) * np.sqrt(periods_per_year)
    return (excess.mean() * periods_per_year) / (ds + 1e-12)

def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())

def calmar(returns: pd.Series) -> float:
    mdd = -max_drawdown(returns)
    if mdd <= 1e-12:
        return np.nan
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    return ann_ret / mdd

def var_historic(returns: pd.Series, level: float = 0.05) -> float:
    return -np.percentile(returns.dropna(), level * 100)

def cvar_historic(returns: pd.Series, level: float = 0.05) -> float:
    thr = np.percentile(returns.dropna(), level * 100)
    tail = returns[returns <= thr]
    return -float(tail.mean()) if len(tail) else -thr

def rolling_var(returns: pd.Series, level: float = 0.05, window: int = 252) -> pd.Series:
    return returns.rolling(window).apply(lambda x: -np.percentile(x, level * 100), raw=False)

# ---- portfolio math ----

def cov_from_corr_sigma(sigmas: np.ndarray, corr: np.ndarray) -> np.ndarray:
    D = np.diag(sigmas)
    return D @ corr @ D

def mean_variance_weights(mu: np.ndarray,
                          cov: np.ndarray,
                          risk_aversion: float = 1.0,
                          long_only: bool = True,
                          l2: float = 1e-6) -> np.ndarray:
    """
    Classic Markowitz: maximize mu^T w - 0.5*λ w^T Σ w - (l2/2)||w||^2.
    Closed form: w ∝ (λ Σ + l2 I)^(-1) mu. Then project if long_only.
    """
    n = len(mu)
    A = risk_aversion * cov + l2 * np.eye(n)
    w = np.linalg.solve(A, mu)
    if long_only:
        w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(n) / n

def target_vol_scale(w: np.ndarray, cov: np.ndarray, target_vol: Optional[float]) -> np.ndarray:
    if target_vol is None:
        return w
    vol = np.sqrt(w @ cov @ w)
    if vol < 1e-12:
        return w
    return w * (target_vol / vol)

# ---- simulation ----

def simulate_paths(mu: np.ndarray,
                   cov: np.ndarray,
                   w: np.ndarray,
                   n_steps: int = 252,
                   n_sims: int = 10_000,
                   dt: float = 1/252,
                   geometric: bool = True,
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate portfolio returns using multivariate normal shocks.
    Returns array of shape (n_sims, n_steps) of portfolio period returns.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    n = len(mu)
    port_rets = np.zeros((n_sims, n_steps), dtype=float)
    for s in range(n_sims):
        shocks = rng.standard_normal((n_steps, n))
        eps = shocks @ L.T
        r = mu * dt + eps * np.sqrt(dt)  # simple diffusion
        p = r @ w
        port_rets[s] = p if not geometric else (np.exp(p) - 1.0)
    return port_rets

def mc_var_cvar(mu: np.ndarray,
                cov: np.ndarray,
                w: np.ndarray,
                horizon_steps: int = 21,
                n_sims: int = 50_000,
                level: float = 0.05,
                seed: Optional[int] = 123) -> Tuple[float, float]:
    """
    Monte Carlo VaR/CVaR over a horizon (sum of period returns).
    """
    paths = simulate_paths(mu, cov, w, n_steps=horizon_steps, n_sims=n_sims, geometric=True, seed=seed)
    agg = paths.sum(axis=1)
    var = -np.percentile(agg, level * 100)
    cvar = -agg[agg <= np.percentile(agg, level * 100)].mean()
    return float(var), float(cvar)

# ---- wrapper ----

def summarize_risk(returns: pd.Series,
                   rf: float = 0.0) -> pd.Series:
    return pd.Series({
        "Sharpe":  sharpe(returns, rf),
        "Sortino": sortino(returns, rf),
        "MaxDD":   max_drawdown(returns),
        "Calmar":  calmar(returns),
        "VaR(5%)": var_historic(returns, 0.05),
        "CVaR(5%)": cvar_historic(returns, 0.05),
    })

# ---- demo ----

if __name__ == "__main__":
    # toy returns
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    r = pd.Series(np.random.normal(0.0005, 0.01, size=len(idx)), index=idx)
    print(summarize_risk(r))

    # toy portfolio
    mu = np.array([0.08, 0.06, 0.10])     # annual drift
    sig = np.array([0.20, 0.15, 0.25])    # annual vol
    corr = np.array([[1.0, 0.3, 0.2],
                     [0.3, 1.0, 0.4],
                     [0.2, 0.4, 1.0]])
    cov = cov_from_corr_sigma(sig, corr)
    w = mean_variance_weights(mu, cov, risk_aversion=3.0, long_only=True)
    w = target_vol_scale(w, cov, target_vol=0.12)
    var5, cvar5 = mc_var_cvar(mu, cov, w, horizon_steps=21, n_sims=20000, level=0.05)
    print("Weights:", np.round(w, 4), "VaR(5%):", round(var5, 4), "CVaR(5%):", round(cvar5, 4))
