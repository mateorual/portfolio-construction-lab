"""Shared synthetic fixtures - the test suite never touches the network."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest

from portfolio_lab.config import load_config
from portfolio_lab.data import MarketData

TICKERS = ["A", "B", "C", "D", "E"]


def random_cov(n: int, seed: int) -> pd.DataFrame:
    """A well-conditioned annualised covariance with heterogeneous vols and correlations."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.05, 0.35, n)
    X = rng.normal(size=(n, n + 3))
    C = X @ X.T
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    corr = 0.7 * corr + 0.3 * np.eye(n)  # keep it comfortably positive definite
    cov = np.outer(vols, vols) * corr
    names = [f"X{i}" for i in range(n)]
    return pd.DataFrame(cov, index=names, columns=names)


@pytest.fixture(params=[0, 1, 2])
def cov(request) -> pd.DataFrame:
    return random_cov(6, request.param)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2015-01-01", periods=900)
    vols = np.array([0.20, 0.15, 0.10, 0.25, 0.05]) / np.sqrt(252)
    drift = np.array([0.08, 0.06, 0.04, 0.10, 0.02]) / 252
    z = rng.standard_t(df=5, size=(len(idx), len(TICKERS))) * np.sqrt(3 / 5)
    return pd.DataFrame(drift + z * vols, index=idx, columns=TICKERS)


@pytest.fixture
def config(tmp_path):
    universe = "\n".join(f"  {t}: {{name: Asset {t}, group: G{i % 2}}}" for i, t in
                         enumerate(TICKERS))
    text = textwrap.dedent("""\
        name: synthetic
        title: Synthetic universe
        universe:
        {universe}
        data:
          start: "2015-01-01"
          end: "2019-01-01"
          risk_free: {{constant: 0.0}}
        backtest:
          lookback_days: 252
          transaction_cost_bps: 10
        strategies:
          - {{id: equal_weight, label: EW, kind: equal_weight}}
          - {{id: min_variance, label: GMV, kind: min_variance}}
          - {{id: mean_variance, label: MV, kind: mean_variance}}
          - {{id: black_litterman, label: BL, kind: black_litterman}}
          - {{id: risk_parity, label: ERC, kind: risk_parity}}
          - {{id: relaxed_risk_parity, label: RRP, kind: relaxed_risk_parity}}
        """).format(universe=universe)
    path = tmp_path / "synthetic.yaml"
    path.write_text(text)
    return load_config(path)


@pytest.fixture
def market_data(synthetic_returns) -> MarketData:
    r = synthetic_returns
    prices = (1 + r).cumprod() * 100
    mkt = pd.DataFrame(1.0 / r.shape[1], index=r.index, columns=r.columns)
    return MarketData(prices=prices, returns=r, risk_free=pd.Series(0.0, index=r.index),
                      market_weights=mkt)
