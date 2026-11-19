"""Experiment configuration.

Every experiment is fully described by one YAML file in ``configs/``. The YAML is
parsed into small frozen dataclasses so the rest of the code never touches raw
dictionaries and typos fail fast at load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    group: str


@dataclass(frozen=True)
class DataConfig:
    start: str
    end: str
    risk_free: dict[str, Any]  # {"ticker": "^IRX"} or {"constant": 0.06}
    market_caps: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    lookback_days: int = 252
    rebalance: str = "monthly"
    transaction_cost_bps: float = 10.0
    periods_per_year: int = 252


@dataclass(frozen=True)
class EstimationConfig:
    covariance: str = "ledoit_wolf"
    ewma_halflife: int = 63
    risk_aversion: float = 2.5
    momentum_lookback: int = 252
    momentum_skip: int = 21


@dataclass(frozen=True)
class ConstraintsConfig:
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass(frozen=True)
class StrategySpec:
    id: str
    label: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentsConfig:
    covariance_ablation: tuple[str, ...] = ("sample", "ledoit_wolf", "ewma")
    cost_sensitivity_bps: tuple[float, ...] = (0, 5, 10, 25, 50)
    bootstrap_samples: int = 2000
    bootstrap_block: int = 21
    resampling_draws: int = 300
    seed: int = 7


@dataclass(frozen=True)
class Config:
    name: str
    title: str
    description: str
    currency: str
    assets: tuple[Asset, ...]
    data: DataConfig
    backtest: BacktestConfig
    estimation: EstimationConfig
    constraints: ConstraintsConfig
    strategies: tuple[StrategySpec, ...]
    experiments: ExperimentsConfig
    source_path: Path | None = None

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def groups(self) -> dict[str, str]:
        return {a.ticker: a.group for a in self.assets}

    @property
    def asset_names(self) -> dict[str, str]:
        return {a.ticker: a.name for a in self.assets}

    @property
    def group_order(self) -> list[str]:
        """Groups in first-appearance order (keeps colours stable across figures)."""
        seen: list[str] = []
        for a in self.assets:
            if a.group not in seen:
                seen.append(a.group)
        return seen

    def strategy(self, strategy_id: str) -> StrategySpec:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        raise KeyError(strategy_id)


def _build(cls, raw: dict[str, Any] | None):
    raw = dict(raw or {})
    known = cls.__dataclass_fields__.keys()
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in raw.items():
        if isinstance(v, list):
            raw[k] = tuple(v)
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    assets = tuple(
        Asset(ticker=t, name=spec["name"], group=spec["group"])
        for t, spec in raw["universe"].items()
    )
    strategies = tuple(
        StrategySpec(id=s["id"], label=s["label"], kind=s["kind"], params=s.get("params", {}))
        for s in raw["strategies"]
    )
    ids = [s.id for s in strategies]
    if len(set(ids)) != len(ids):
        raise ValueError("Strategy ids must be unique")

    data_raw = dict(raw["data"])
    return Config(
        name=raw["name"],
        title=raw["title"],
        description=raw.get("description", ""),
        currency=raw.get("currency", "USD"),
        assets=assets,
        data=DataConfig(**data_raw),
        backtest=_build(BacktestConfig, raw.get("backtest")),
        estimation=_build(EstimationConfig, raw.get("estimation")),
        constraints=_build(ConstraintsConfig, raw.get("constraints")),
        strategies=strategies,
        experiments=_build(ExperimentsConfig, raw.get("experiments")),
        source_path=path,
    )
