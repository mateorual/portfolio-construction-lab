"""Performance, diversification and statistical-inference metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizers import risk_contributions


# --------------------------------------------------------------------------- #
# Return-based performance
# --------------------------------------------------------------------------- #
def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def sharpe_ratio(returns: np.ndarray | pd.Series, rf: np.ndarray | pd.Series | float = 0.0,
                 periods: int = 252) -> float:
    """Annualised Sharpe ratio of *excess* returns (arithmetic mean / std)."""
    ex = np.asarray(returns, dtype=float) - np.asarray(rf, dtype=float)
    sd = ex.std(ddof=1)
    return float(np.sqrt(periods) * ex.mean() / sd) if sd > 0 else np.nan


def performance_summary(returns: pd.Series, rf: pd.Series, periods: int = 252) -> dict[str, float]:
    rf = rf.reindex(returns.index).fillna(0.0)
    ex = returns - rf
    n_years = len(returns) / periods
    wealth = float((1.0 + returns).prod())
    cagr = wealth ** (1.0 / n_years) - 1.0
    vol = float(returns.std(ddof=1) * np.sqrt(periods))
    downside = float(np.sqrt(np.mean(np.minimum(ex.values, 0.0) ** 2)) * np.sqrt(periods))
    mdd = float(drawdown(returns).min())
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe_ratio(returns, rf, periods),
        "Sortino": float(ex.mean() * periods / downside) if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# Portfolio-structure diagnostics
# --------------------------------------------------------------------------- #
def effective_number(shares: np.ndarray) -> float:
    """Inverse Herfindahl index: N for equal shares, 1 for full concentration."""
    s = np.asarray(shares, dtype=float)
    s = s / s.sum()
    return float(1.0 / np.sum(s**2))


def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    """Choueifaty & Coignard (2008): weighted average vol / portfolio vol (>= 1)."""
    vols = np.sqrt(np.diag(cov))
    return float((w @ vols) / np.sqrt(w @ cov @ w))


def structure_diagnostics(target_weights: pd.DataFrame, traded: pd.Series,
                          covariances: dict[pd.Timestamp, pd.DataFrame],
                          periods: int = 252) -> dict[str, float]:
    """Averages over rebalance dates of ex-ante portfolio-structure statistics.

    Risk statistics use the covariance estimate available *at* each rebalance.
    """
    n_eff_w, n_eff_rc, max_w, dr = [], [], [], []
    for d, w in target_weights.iterrows():
        S = covariances[d].loc[w.index, w.index].values
        wv = w.values
        n_eff_w.append(effective_number(wv))
        n_eff_rc.append(effective_number(np.clip(risk_contributions(wv, S), 1e-12, None)))
        max_w.append(wv.max())
        dr.append(diversification_ratio(wv, S))
    years = len(traded) / periods
    return {
        "Turnover (1-way, p.a.)": float(0.5 * traded.sum() / years),
        "Effective N (weights)": float(np.mean(n_eff_w)),
        "Effective N (risk)": float(np.mean(n_eff_rc)),
        "Avg max weight": float(np.mean(max_w)),
        "Diversification ratio": float(np.mean(dr)),
    }


def group_shares(weights: pd.DataFrame, groups: dict[str, str], order: list[str]) -> pd.DataFrame:
    """Aggregate per-asset weights (or risk shares) into asset groups."""
    return weights.T.groupby(pd.Series(groups)).sum().T.reindex(columns=order, fill_value=0.0)


def risk_share_path(target_weights: pd.DataFrame,
                    covariances: dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
    """Ex-ante relative risk contributions at each rebalance date."""
    rows = {}
    for d, w in target_weights.iterrows():
        S = covariances[d].loc[w.index, w.index].values
        rows[d] = risk_contributions(w.values, S)
    return pd.DataFrame(rows, index=target_weights.columns).T


# --------------------------------------------------------------------------- #
# Inference: circular block bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_indices(n: int, block: int, n_samples: int, seed: int) -> np.ndarray:
    """Circular block bootstrap (Politis & Romano, 1992) index matrix, shape (n_samples, n).

    Resampling blocks of consecutive days preserves volatility clustering and
    short-range autocorrelation that an i.i.d. bootstrap would destroy.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_samples, -1)[:, :n]


def bootstrap_sharpe(returns: pd.DataFrame, rf: pd.Series, benchmark: str, block: int = 21,
                     n_samples: int = 2000, seed: int = 7, periods: int = 252,
                     alpha: float = 0.05) -> pd.DataFrame:
    """Sharpe ratio confidence intervals and Sharpe differences vs ``benchmark``.

    All strategies are resampled with the *same* index draws, so the paired
    differences keep their cross-correlation.
    """
    ex = returns.sub(rf.reindex(returns.index).fillna(0.0), axis=0).values
    idx = block_bootstrap_indices(len(ex), block, n_samples, seed)
    sr = np.empty((n_samples, ex.shape[1]))
    for start in range(0, n_samples, 100):  # chunked to bound memory
        sample = ex[idx[start:start + 100]]  # (chunk, n, k)
        sr[start:start + 100] = (np.sqrt(periods) * sample.mean(axis=1)
                                 / sample.std(axis=1, ddof=1))
    b = list(returns.columns).index(benchmark)
    diff = sr - sr[:, [b]]
    point = np.sqrt(periods) * ex.mean(axis=0) / ex.std(axis=0, ddof=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = pd.DataFrame({
        "Sharpe": point,
        "Sharpe CI low": np.percentile(sr, lo, axis=0),
        "Sharpe CI high": np.percentile(sr, hi, axis=0),
        "dSharpe vs benchmark": point - point[b],
        "dSharpe CI low": np.percentile(diff, lo, axis=0),
        "dSharpe CI high": np.percentile(diff, hi, axis=0),
        # one-sided: share of resamples where the strategy does NOT beat the benchmark
        "p (dSharpe <= 0)": (diff <= 0).mean(axis=0),
    }, index=returns.columns)
    out.loc[benchmark, ["dSharpe CI low", "dSharpe CI high", "p (dSharpe <= 0)"]] = np.nan
    return out
