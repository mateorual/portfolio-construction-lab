"""Strategies: map a set of point-in-time estimates to target portfolio weights.

At each rebalance date the backtester builds **one** :class:`Estimates` object
from data up to and including that date, and every strategy consumes the same
object. That guarantees the comparison differs only in the portfolio
construction rule - not in data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import black_litterman as bl
from . import estimators as est
from . import optimizers as opt
from .config import Config, StrategySpec


@dataclass
class Estimates:
    date: pd.Timestamp
    returns: pd.DataFrame  # the look-back window of daily returns
    cov: pd.DataFrame  # annualised covariance (configured estimator)
    mu_hist: pd.Series  # annualised sample mean
    momentum: pd.Series  # 12-1 momentum score
    market_weights: pd.Series  # reference portfolio for the BL prior


def make_estimates(returns_window: pd.DataFrame, market_weights: pd.Series, cfg: Config,
                   covariance_method: str | None = None) -> Estimates:
    e = cfg.estimation
    periods = cfg.backtest.periods_per_year
    cov = est.estimate_covariance(returns_window, covariance_method or e.covariance, periods,
                                  e.ewma_halflife)
    return Estimates(
        date=returns_window.index[-1],
        returns=returns_window,
        cov=cov,
        mu_hist=est.historical_mean(returns_window, periods),
        momentum=est.momentum_score(returns_window, e.momentum_lookback, e.momentum_skip),
        market_weights=market_weights.reindex(returns_window.columns),
    )


class Strategy:
    """Base class. Subclasses implement :meth:`target_weights`."""

    def __init__(self, spec: StrategySpec, cfg: Config):
        self.spec = spec
        self.cfg = cfg
        self.params: dict[str, Any] = dict(spec.params)
        self.lo = cfg.constraints.min_weight
        self.hi = cfg.constraints.max_weight

    @property
    def id(self) -> str:
        return self.spec.id

    def target_weights(self, e: Estimates) -> pd.Series:  # pragma: no cover - abstract
        raise NotImplementedError


class EqualWeight(Strategy):
    def target_weights(self, e: Estimates) -> pd.Series:
        n = e.cov.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=e.cov.index)


class MinVariance(Strategy):
    def target_weights(self, e: Estimates) -> pd.Series:
        return opt.min_variance(e.cov, self.lo, self.hi)


class MeanVariance(Strategy):
    """Markowitz with the sample mean as mu - the 'estimation-error maximiser' benchmark."""

    def target_weights(self, e: Estimates) -> pd.Series:
        delta = self.params.get("risk_aversion", self.cfg.estimation.risk_aversion)
        return opt.mean_variance(e.mu_hist, e.cov, delta, self.lo, self.hi)


class BlackLitterman(Strategy):
    """Market-implied prior + momentum views -> posterior -> mean-variance."""

    def posterior(self, e: Estimates) -> bl.BlackLittermanResult:
        p = self.params
        return bl.black_litterman_momentum(
            cov=e.cov,
            market_weights=e.market_weights,
            momentum=e.momentum,
            risk_aversion=p.get("risk_aversion", self.cfg.estimation.risk_aversion),
            tau=p.get("tau", 0.05),
            information_coefficient=p.get("information_coefficient", 0.1),
            confidence=p.get("view_confidence", 1.0),
        )

    def target_weights(self, e: Estimates) -> pd.Series:
        res = self.posterior(e)
        delta = self.params.get("risk_aversion", self.cfg.estimation.risk_aversion)
        return opt.mean_variance(res.posterior, res.posterior_cov, delta, self.lo, self.hi)


class RiskParity(Strategy):
    def target_weights(self, e: Estimates) -> pd.Series:
        return opt.risk_parity(e.cov)


class RelaxedRiskParity(Strategy):
    def target_weights(self, e: Estimates) -> pd.Series:
        mu = e.mu_hist
        if self.params.get("mu_source", "historical") == "black_litterman":
            bl_spec = StrategySpec("tmp", "tmp", "black_litterman", self.params.get("bl", {}))
            mu = BlackLitterman(bl_spec, self.cfg).posterior(e).posterior
        return opt.relaxed_risk_parity(
            mu, e.cov,
            return_premium=self.params.get("return_premium", 0.25),
            lam=self.params.get("lam", 0.0),
            min_weight=self.lo, max_weight=self.hi,
        )


REGISTRY: dict[str, type[Strategy]] = {
    "equal_weight": EqualWeight,
    "min_variance": MinVariance,
    "mean_variance": MeanVariance,
    "black_litterman": BlackLitterman,
    "risk_parity": RiskParity,
    "relaxed_risk_parity": RelaxedRiskParity,
}


def build_strategy(spec: StrategySpec, cfg: Config) -> Strategy:
    try:
        return REGISTRY[spec.kind](spec, cfg)
    except KeyError as exc:
        raise ValueError(f"Unknown strategy kind '{spec.kind}'. Known: {sorted(REGISTRY)}") from exc


def build_strategies(cfg: Config) -> list[Strategy]:
    return [build_strategy(s, cfg) for s in cfg.strategies]
