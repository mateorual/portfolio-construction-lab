"""Covariance and expected-return estimators.

All functions take a window of *daily simple returns* and return **annualised**
quantities, so every optimizer downstream works in the same units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

COVARIANCE_METHODS = ("sample", "ledoit_wolf", "ewma")


def sample_covariance(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return returns.cov() * periods


def ledoit_wolf_covariance(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Ledoit & Wolf (2004) shrinkage towards a scaled identity.

    Shrinkage pulls extreme eigenvalues of the sample covariance towards their
    mean, which is exactly what stabilises inverse-covariance-driven optimizers.
    """
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_ * periods
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_covariance(returns: pd.DataFrame, halflife: int = 63, periods: int = 252) -> pd.DataFrame:
    """Exponentially-weighted covariance (RiskMetrics style) with a given half-life in days."""
    x = returns.values
    n = len(x)
    decay = 0.5 ** (1.0 / halflife)
    w = decay ** np.arange(n - 1, -1, -1)
    w /= w.sum()
    mu = w @ x
    xc = x - mu
    cov = (xc * w[:, None]).T @ xc / (1.0 - np.sum(w**2))  # unbiased weighting
    return pd.DataFrame(cov * periods, index=returns.columns, columns=returns.columns)


def estimate_covariance(returns: pd.DataFrame, method: str, periods: int = 252,
                        ewma_halflife: int = 63) -> pd.DataFrame:
    if method == "sample":
        cov = sample_covariance(returns, periods)
    elif method == "ledoit_wolf":
        cov = ledoit_wolf_covariance(returns, periods)
    elif method == "ewma":
        cov = ewma_covariance(returns, ewma_halflife, periods)
    else:
        raise ValueError(f"Unknown covariance method '{method}'. Use one of {COVARIANCE_METHODS}")
    # enforce exact symmetry (numerical noise breaks some solvers' PSD checks)
    return (cov + cov.T) / 2.0


def historical_mean(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    """Annualised arithmetic mean - the textbook (and notoriously noisy) estimate of mu."""
    return returns.mean() * periods


def momentum_score(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Cross-sectional time-series momentum: compounded return from t-lookback to t-skip.

    Skipping the most recent month avoids the well-documented short-term
    reversal effect (Jegadeesh & Titman, 1993).
    """
    if len(returns) < lookback:
        raise ValueError(f"Need {lookback} observations for momentum, got {len(returns)}")
    window = returns.iloc[-lookback: len(returns) - skip] if skip else returns.iloc[-lookback:]
    return (1.0 + window).prod() - 1.0


def is_positive_definite(cov: np.ndarray, tol: float = 0.0) -> bool:
    return bool(np.linalg.eigvalsh(cov).min() > tol)
