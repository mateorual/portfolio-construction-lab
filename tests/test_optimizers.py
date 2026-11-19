import numpy as np
import pandas as pd
import pytest

from portfolio_lab import optimizers as opt


def _check_budget(w: pd.Series, lo: float = 0.0, hi: float = 1.0) -> None:
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= lo - 1e-8).all() and (w <= hi + 1e-6).all()


# --------------------------------------------------------------------------- #
# Risk parity
# --------------------------------------------------------------------------- #
def test_erc_equalises_risk_contributions(cov):
    w = opt.risk_parity(cov)
    _check_budget(w)
    rc = opt.risk_contributions(w, cov)
    np.testing.assert_allclose(rc, 1 / len(cov), atol=1e-5)


def test_erc_with_daily_units_is_not_stuck_at_equal_weight(cov):
    """Regression test: a badly scaled (daily) covariance must give the same answer."""
    w_annual = opt.risk_parity(cov)
    w_daily = opt.risk_parity(cov / 252)
    np.testing.assert_allclose(w_annual, w_daily, atol=1e-6)
    assert not np.allclose(w_daily, 1 / len(cov), atol=1e-3)


def test_erc_diagonal_covariance_is_inverse_volatility():
    vols = np.array([0.1, 0.2, 0.4])
    cov = pd.DataFrame(np.diag(vols**2), index=list("abc"), columns=list("abc"))
    w = opt.risk_parity(cov)
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_risk_budgets_are_respected(cov):
    b = np.linspace(1, 3, len(cov))
    b = b / b.sum()
    w = opt.risk_parity(cov, budgets=b)
    np.testing.assert_allclose(opt.risk_contributions(w, cov), b, atol=1e-5)


def test_absolute_risk_contributions_sum_to_volatility(cov):
    w = np.full(len(cov), 1 / len(cov))
    rc = opt.risk_contributions(w, cov, relative=False)
    assert rc.sum() == pytest.approx(np.sqrt(w @ cov.values @ w))


# --------------------------------------------------------------------------- #
# Markowitz family
# --------------------------------------------------------------------------- #
def test_min_variance_matches_closed_form_when_unconstrained_solution_is_long_only():
    vols = np.array([0.10, 0.15, 0.20, 0.25])
    corr = np.full((4, 4), 0.2) + 0.8 * np.eye(4)
    S = np.outer(vols, vols) * corr
    cov = pd.DataFrame(S, index=list("abcd"), columns=list("abcd"))
    inv = np.linalg.inv(S) @ np.ones(4)
    closed_form = inv / inv.sum()
    assert (closed_form > 0).all()
    np.testing.assert_allclose(opt.min_variance(cov), closed_form, atol=1e-6)


def test_min_variance_is_minimal(cov):
    w = opt.min_variance(cov).values
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.dirichlet(np.ones(len(cov)))
        assert w @ cov.values @ w <= x @ cov.values @ x + 1e-10


def test_mean_variance_respects_caps_and_tends_to_min_variance(cov):
    mu = pd.Series(np.linspace(0.02, 0.12, len(cov)), index=cov.index)
    capped = opt.mean_variance(mu, cov, risk_aversion=1.0, max_weight=0.3)
    _check_budget(capped, hi=0.3)
    very_averse = opt.mean_variance(mu, cov, risk_aversion=1e6)
    np.testing.assert_allclose(very_averse, opt.min_variance(cov), atol=1e-4)


def test_frontier_portfolio_hits_target(cov):
    mu = pd.Series(np.linspace(0.02, 0.12, len(cov)), index=cov.index)
    w = opt.frontier_portfolio(mu, cov, 0.08)
    assert mu @ w >= 0.08 - 1e-6


# --------------------------------------------------------------------------- #
# Relaxed risk parity
# --------------------------------------------------------------------------- #
def test_rrp_with_zero_premium_is_erc(cov):
    mu = pd.Series(np.linspace(0.02, 0.12, len(cov)), index=cov.index)
    np.testing.assert_allclose(opt.relaxed_risk_parity(mu, cov, return_premium=0.0),
                               opt.risk_parity(cov), atol=1e-4)


def _equicorrelated(n: int = 6, rho: float = 0.3) -> pd.DataFrame:
    vols = np.linspace(0.05, 0.30, n)
    S = np.outer(vols, vols) * (np.full((n, n), rho) + (1 - rho) * np.eye(n))
    names = [f"X{i}" for i in range(n)]
    return pd.DataFrame(S, index=names, columns=names)


def test_rrp_meets_return_target_and_trades_away_parity():
    cov = _equicorrelated()  # positive correlations: every target below r_max is feasible
    mu = pd.Series(np.linspace(0.02, 0.12, len(cov)), index=cov.index)
    erc = opt.risk_parity(cov)
    r_erc, r_max = float(mu @ erc), float(mu.max())
    prev_ret, prev_disp = r_erc, 0.0
    for kappa in (0.2, 0.4, 0.6):
        w = opt.relaxed_risk_parity(mu, cov, return_premium=kappa)
        _check_budget(w)
        ret = float(mu @ w)
        assert ret >= r_erc + kappa * (r_max - r_erc) - 1e-6
        disp = float(np.std(opt.risk_contributions(w, cov)))
        assert ret >= prev_ret - 1e-8 and disp >= prev_disp - 1e-6
        prev_ret, prev_disp = ret, disp


def test_rrp_degrades_gracefully_when_target_is_infeasible(cov):
    """With negative correlations the cone constraints can cap attainable return."""
    mu = pd.Series(np.linspace(0.02, 0.12, len(cov)), index=cov.index)
    r_erc = float(mu @ opt.risk_parity(cov))
    prev = r_erc
    for kappa in (0.2, 0.5, 0.9):
        w = opt.relaxed_risk_parity(mu, cov, return_premium=kappa)
        _check_budget(w)
        assert float(mu @ w) >= prev - 1e-6  # never worse than a lower premium
        prev = float(mu @ w)


def test_rrp_rejects_degenerate_premium(cov):
    mu = pd.Series(0.05, index=cov.index)
    with pytest.raises(ValueError):
        opt.relaxed_risk_parity(mu, cov, return_premium=1.0)
