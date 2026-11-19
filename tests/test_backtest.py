import numpy as np
import pandas as pd
import pytest

from portfolio_lab.backtest import (
    compute_estimates,
    compute_targets,
    rebalance_dates,
    run_backtests,
    simulate,
)
from portfolio_lab.strategies import Strategy, build_strategies


def test_rebalance_dates_are_month_ends_after_lookback(synthetic_returns):
    idx = synthetic_returns.index
    dates = rebalance_dates(idx, lookback=252)
    assert dates[0] >= idx[251]
    assert dates[-1] < idx[-1]
    for d in dates:  # last trading day of its month
        later = idx[idx > d]
        assert len(later) == 0 or later[0].month != d.month


def test_simulate_drift_turnover_and_costs():
    idx = pd.bdate_range("2020-01-30", periods=4)
    r = pd.DataFrame({"a": [0.0, 0.10, 0.0, 0.0], "b": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    targets = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=[idx[0], idx[2]])
    res = simulate(r, targets, cost_bps=10, strategy_id="x")
    # day 1: +10% on half the book
    assert res.gross_returns.iloc[0] == pytest.approx(0.05)
    # drifted weights before the day-2 rebalance: 0.55/1.05 and 0.5/1.05
    drifted = np.array([0.55, 0.5]) / 1.05
    expected_traded = np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert res.traded.loc[idx[2]] == pytest.approx(expected_traded)
    net = res.net_returns()
    assert net.loc[idx[2]] == pytest.approx(-expected_traded * 10 / 1e4)
    assert res.net_returns(0).equals(res.gross_returns.rename("x"))


class _Recorder(Strategy):
    """Strategy that records the last date it was allowed to see."""

    seen: list

    def target_weights(self, e):
        self.seen.append((e.date, e.returns.index.max()))
        n = e.cov.shape[0]
        return pd.Series(1.0 / n, index=e.cov.index)


def test_estimates_never_use_future_data(market_data, config):
    rec = _Recorder(config.strategies[0], config)
    rec.seen = []
    dates = rebalance_dates(market_data.returns.index, config.backtest.lookback_days)
    compute_targets(rec, compute_estimates(market_data, config, dates))
    assert rec.seen and all(last_seen <= d for d, last_seen in rec.seen)


def test_future_returns_do_not_change_past_decisions(market_data, config):
    strategies = build_strategies(config)
    base = run_backtests(market_data, strategies, config)
    cut = market_data.returns.index[600]
    shocked = market_data.returns.copy()
    shocked.loc[shocked.index > cut] *= -3.0  # rewrite the future
    market_data.returns = shocked
    alt = run_backtests(market_data, strategies, config)
    for sid in base:
        a = base[sid].target_weights.loc[:cut]
        b = alt[sid].target_weights.loc[:cut]
        pd.testing.assert_frame_equal(a, b)


def test_full_backtest_runs_for_every_strategy(market_data, config):
    results = run_backtests(market_data, build_strategies(config), config)
    assert set(results) == {s.id for s in config.strategies}
    for r in results.values():
        assert not r.failures
        assert np.allclose(r.target_weights.sum(axis=1), 1.0)
        assert r.returns.notna().all()
