"""Monthly-rebalancing backtest engine.

Timing convention (no look-ahead by construction):

1. At the close of rebalance date *t* the strategy sees returns up to and
   including *t* (a fixed look-back window).
2. It trades from its *drifted* weights to the new target at the close of *t*;
   the cost ``bps x sum|delta w|`` is charged to day *t*.
3. The new weights earn returns from *t+1* onwards and drift with prices until
   the next rebalance.

The engine runs in two phases: (i) compute every strategy's targets on the
shared estimates of each rebalance date, (ii) simulate the daily path. Because
the gross return path and the traded notional are stored separately, net
returns for any other cost level are a one-line recomputation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketData
from .optimizers import OptimizationError
from .strategies import Estimates, Strategy, make_estimates

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    strategy_id: str
    gross_returns: pd.Series  # daily, before costs
    traded: pd.Series  # daily sum |delta w| (non-zero on rebalance days only)
    target_weights: pd.DataFrame  # one row per rebalance date
    cost_bps: float
    failures: list[pd.Timestamp] = field(default_factory=list)

    def net_returns(self, cost_bps: float | None = None) -> pd.Series:
        c = self.cost_bps if cost_bps is None else cost_bps
        return (self.gross_returns - self.traded * c / 1e4).rename(self.strategy_id)

    @property
    def returns(self) -> pd.Series:
        return self.net_returns()


def rebalance_dates(index: pd.DatetimeIndex, lookback: int, frequency: str = "monthly"
                    ) -> pd.DatetimeIndex:
    """Last trading day of each period, once ``lookback`` observations are available."""
    period = {"monthly": "M", "quarterly": "Q"}.get(frequency)
    if period is None:
        raise ValueError(f"Unsupported rebalance frequency '{frequency}'")
    s = index.to_series()
    ends = pd.DatetimeIndex(s.groupby(index.to_period(period)).max().values)
    eligible = index[lookback - 1:]
    ends = ends[(ends >= eligible[0]) & (ends < index[-1])]  # need >= 1 day to hold
    return ends


def compute_estimates(data: MarketData, cfg: Config, dates: Sequence[pd.Timestamp],
                      covariance_method: str | None = None) -> dict[pd.Timestamp, Estimates]:
    lookback = cfg.backtest.lookback_days
    pos = data.returns.index.get_indexer(dates)
    out = {}
    for d, i in zip(dates, pos, strict=True):
        window = data.returns.iloc[i - lookback + 1: i + 1]
        assert window.index[-1] == d and len(window) == lookback
        out[d] = make_estimates(window, data.market_weights.loc[d], cfg, covariance_method)
    return out


def compute_targets(strategy: Strategy, estimates: dict[pd.Timestamp, Estimates]
                    ) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Target weights per rebalance date; on solver failure keep the previous target."""
    rows, failures, prev = {}, [], None
    for d, e in estimates.items():
        try:
            w = strategy.target_weights(e)
        except OptimizationError as exc:
            failures.append(d)
            log.warning("%s: optimisation failed on %s (%s) - holding previous target",
                        strategy.id, d.date(), exc)
            w = prev if prev is not None else pd.Series(1.0 / e.cov.shape[0], index=e.cov.index)
        rows[d] = w
        prev = w
    return pd.DataFrame(rows).T.rename_axis("date"), failures


def simulate(returns: pd.DataFrame, targets: pd.DataFrame, cost_bps: float,
             strategy_id: str) -> BacktestResult:
    """Daily path with weight drift. The initial buy-in is not charged (identical for all)."""
    tickers = list(returns.columns)
    targets = targets[tickers]
    first = targets.index[0]
    R = returns.loc[returns.index > first]
    rebal = {d: targets.loc[d].values for d in targets.index[1:]}

    w = targets.loc[first].values.astype(float)
    gross = np.empty(len(R))
    traded = np.zeros(len(R))
    for k, (date, r) in enumerate(zip(R.index, R.values, strict=True)):
        gp = float(w @ r)
        gross[k] = gp
        w = w * (1.0 + r) / (1.0 + gp)  # drift with prices
        tgt = rebal.get(date)
        if tgt is not None:
            traded[k] = np.abs(tgt - w).sum()
            w = tgt.astype(float)
    return BacktestResult(
        strategy_id=strategy_id,
        gross_returns=pd.Series(gross, index=R.index, name=strategy_id),
        traded=pd.Series(traded, index=R.index, name=strategy_id),
        target_weights=targets,
        cost_bps=cost_bps,
    )


def run_backtests(data: MarketData, strategies: Sequence[Strategy], cfg: Config,
                  covariance_method: str | None = None,
                  estimates: dict[pd.Timestamp, Estimates] | None = None
                  ) -> dict[str, BacktestResult]:
    if estimates is None:
        dates = rebalance_dates(data.returns.index, cfg.backtest.lookback_days,
                                cfg.backtest.rebalance)
        log.info("Rebalances: %d (%s -> %s), covariance=%s", len(dates), dates[0].date(),
                 dates[-1].date(), covariance_method or cfg.estimation.covariance)
        estimates = compute_estimates(data, cfg, dates, covariance_method)
    results = {}
    for s in strategies:
        targets, failures = compute_targets(s, estimates)
        res = simulate(data.returns, targets, cfg.backtest.transaction_cost_bps, s.id)
        res.failures = failures
        results[s.id] = res
    return results
