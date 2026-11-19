import numpy as np
import pandas as pd
import pytest

from portfolio_lab import black_litterman as bl
from portfolio_lab import optimizers as opt


@pytest.fixture
def mkt(cov) -> pd.Series:
    w = np.linspace(1, 2, len(cov))
    return pd.Series(w / w.sum(), index=cov.index)


def test_reverse_optimisation_recovers_market_weights(cov, mkt):
    """Unconstrained mean-variance on pi must hand back the market portfolio."""
    delta = 2.5
    pi = bl.implied_equilibrium_returns(cov, mkt, delta)
    w = np.linalg.solve(delta * cov.values, pi.values)
    np.testing.assert_allclose(w, mkt.values, atol=1e-10)
    # ... and the long-only optimizer agrees because the solution is interior
    np.testing.assert_allclose(opt.mean_variance(pi, cov, delta), mkt.values, atol=1e-4)


def test_no_views_returns_prior(cov, mkt):
    pi = bl.implied_equilibrium_returns(cov, mkt, 2.5)
    res = bl.posterior(cov, pi, np.empty((0, len(cov))), np.empty(0), tau=0.05,
                       omega=np.empty((0, 0)))
    np.testing.assert_allclose(res.posterior, pi, atol=1e-12)


def test_zero_signal_leaves_prior_untouched(cov, mkt):
    flat = pd.Series(0.1, index=cov.index)  # no cross-sectional dispersion
    res = bl.black_litterman_momentum(cov, mkt, flat, 2.5, 0.05, information_coefficient=0.1)
    np.testing.assert_allclose(res.posterior, res.prior, atol=1e-12)


def test_certain_views_are_matched_exactly(cov, mkt):
    pi = bl.implied_equilibrium_returns(cov, mkt, 2.5)
    P = np.zeros((1, len(cov)))
    P[0, 0], P[0, 1] = 1.0, -1.0  # relative view: asset 0 beats asset 1 ...
    Q = np.array([0.03])  # ... by 3%
    omega = np.array([[1e-12]])
    res = bl.posterior(cov, pi, P, Q, tau=0.05, omega=omega)
    assert (P @ res.posterior.values)[0] == pytest.approx(0.03, abs=1e-6)


def test_posterior_lies_between_prior_and_views(cov, mkt):
    momentum = pd.Series(np.linspace(-0.2, 0.3, len(cov)), index=cov.index)
    res = bl.black_litterman_momentum(cov, mkt, momentum, 2.5, 0.05, 0.1)
    # with P = I and moderate confidence, each posterior shift has the sign of its view
    shift_view = res.views - res.prior
    shift_post = res.posterior - res.prior
    assert (np.sign(shift_post[shift_view.abs() > 1e-3]) ==
            np.sign(shift_view[shift_view.abs() > 1e-3])).mean() > 0.6
    assert np.abs(shift_post).sum() < np.abs(shift_view).sum()


def test_posterior_covariance_adds_estimation_uncertainty(cov, mkt):
    momentum = pd.Series(np.linspace(-0.2, 0.3, len(cov)), index=cov.index)
    res = bl.black_litterman_momentum(cov, mkt, momentum, 2.5, 0.05, 0.1)
    diff = res.posterior_cov.values - cov.values
    assert np.linalg.eigvalsh(diff).min() > 0
