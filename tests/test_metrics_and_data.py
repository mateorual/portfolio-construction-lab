import numpy as np
import pandas as pd
import pytest

from portfolio_lab import estimators as est
from portfolio_lab import metrics as M
from portfolio_lab.data import clean_prices, mask_stale_prices


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20, 0.50])
    dd = M.drawdown(r)
    assert dd.min() == pytest.approx(-0.50)
    assert dd.iloc[-1] == pytest.approx(1.1 * 0.5 * 1.2 * 1.5 / 1.1 - 1)


def test_sharpe_scales_with_sqrt_time():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0004, 0.01, 5000)
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert M.sharpe_ratio(r) == pytest.approx(expected)


def test_effective_number():
    assert M.effective_number(np.ones(10)) == pytest.approx(10)
    assert M.effective_number(np.array([1, 0, 0])) == pytest.approx(1)


def test_diversification_ratio_is_one_for_single_asset_and_perfect_correlation():
    S = np.outer([0.1, 0.2], [0.1, 0.2])  # correlation one
    assert M.diversification_ratio(np.array([0.5, 0.5]), S) == pytest.approx(1.0)


def test_block_bootstrap_indices_are_circular_blocks():
    idx = M.block_bootstrap_indices(n=50, block=10, n_samples=3, seed=0)
    assert idx.shape == (3, 50)
    steps = np.diff(idx[:, :10], axis=1) % 50
    assert (steps == 1).all()


def test_bootstrap_is_reproducible_and_benchmark_difference_is_zero():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, (600, 3)), index=idx, columns=["b", "x", "y"])
    rf = pd.Series(0.0, index=idx)
    a = M.bootstrap_sharpe(rets, rf, "b", n_samples=200, seed=3)
    b = M.bootstrap_sharpe(rets, rf, "b", n_samples=200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["b", "dSharpe vs benchmark"] == 0
    assert (a["Sharpe CI low"] < a["Sharpe"]).all() and (a["Sharpe"] < a["Sharpe CI high"]).all()


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def test_ledoit_wolf_improves_conditioning(synthetic_returns):
    short = synthetic_returns.iloc[:30]  # few observations -> ill-conditioned sample cov
    s = np.linalg.cond(est.sample_covariance(short).values)
    lw = np.linalg.cond(est.ledoit_wolf_covariance(short).values)
    assert lw < s


def test_ewma_with_long_halflife_approaches_sample(synthetic_returns):
    s = est.sample_covariance(synthetic_returns).values
    e = est.ewma_covariance(synthetic_returns, halflife=10**7).values
    np.testing.assert_allclose(e, s, rtol=1e-3, atol=1e-6)


def test_momentum_skips_most_recent_month(synthetic_returns):
    r = synthetic_returns.copy()
    base = est.momentum_score(r, 252, 21)
    r.iloc[-21:] = 0.5  # huge moves in the skipped month must not matter
    pd.testing.assert_series_equal(base, est.momentum_score(r, 252, 21))


# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
def test_stale_prices_are_masked_and_sample_starts_after_them():
    idx = pd.bdate_range("2020-01-01", periods=40)
    good = pd.Series(np.linspace(100, 140, 40), index=idx)
    stale = good.copy()
    stale.iloc[:15] = 50.0  # vendor back-fill
    prices = pd.DataFrame({"good": good, "stale": stale})
    masked = mask_stale_prices(prices)
    assert masked["stale"].iloc[:15].isna().all()
    assert masked["good"].notna().all()
    cleaned = clean_prices(prices)
    assert cleaned.index[0] == idx[15]
