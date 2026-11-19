"""Market data: download once from Yahoo Finance, then serve from a local parquet cache.

The cache makes every run after the first deterministic and offline, which is
what you want for research: results only change when you *decide* to refresh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

MAX_FFILL_DAYS = 5  # tolerate short gaps (exchange holidays, missing prints)
STALE_RUN_DAYS = 5  # >= 5 identical consecutive closes => vendor back-fill, not trading
EXTREME_DAILY_MOVE = 0.40


@dataclass
class MarketData:
    prices: pd.DataFrame  # adjusted close, columns = tickers
    returns: pd.DataFrame  # simple daily returns
    risk_free: pd.Series  # daily simple risk-free return, aligned to `returns`
    market_weights: pd.DataFrame  # reference ("market") portfolio weights per day


# --------------------------------------------------------------------------- #
# Download helpers (network)
# --------------------------------------------------------------------------- #
def _download_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No data returned for: {missing}")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return close[tickers]


def _download_shares_outstanding(tickers: list[str]) -> pd.Series:
    import yfinance as yf

    shares = {}
    for t in tickers:
        value = None
        try:
            value = yf.Ticker(t).fast_info.get("shares")
        except Exception:  # noqa: BLE001 - yfinance raises a zoo of exceptions
            value = None
        if not value:
            try:
                value = yf.Ticker(t).info.get("sharesOutstanding")
            except Exception:  # noqa: BLE001
                value = None
        if not value:
            raise RuntimeError(f"Could not obtain shares outstanding for {t}")
        shares[t] = float(value)
    return pd.Series(shares, name="shares_outstanding")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
def _cache_dir(cfg: Config, root: Path) -> Path:
    d = root / cfg.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_prices(cfg: Config, data_root: Path, refresh: bool = False) -> pd.DataFrame:
    path = _cache_dir(cfg, data_root) / "prices.parquet"
    if path.exists() and not refresh:
        prices = pd.read_parquet(path)
        if set(cfg.tickers) <= set(prices.columns):
            return prices[cfg.tickers]
        log.info("Cached prices miss some tickers; re-downloading")
    log.info("Downloading %d tickers from Yahoo Finance", len(cfg.tickers))
    prices = _download_close(cfg.tickers, cfg.data.start, cfg.data.end)
    prices.to_parquet(path)
    return prices


def load_risk_free(cfg: Config, index: pd.DatetimeIndex, data_root: Path,
                   refresh: bool = False) -> pd.Series:
    """Daily simple risk-free return aligned to ``index``.

    ``{"ticker": "^IRX"}`` uses the 13-week T-bill yield (quoted in percent p.a.);
    ``{"constant": 0.06}`` uses a flat annual rate.
    """
    spec = cfg.data.risk_free
    if "constant" in spec:
        annual = pd.Series(float(spec["constant"]), index=index)
    else:
        path = _cache_dir(cfg, data_root) / "risk_free.parquet"
        if path.exists() and not refresh:
            raw = pd.read_parquet(path).iloc[:, 0]
        else:
            raw = _download_close([spec["ticker"]], cfg.data.start, cfg.data.end).iloc[:, 0]
            raw.to_frame("yield_pct").to_parquet(path)
        annual = raw.reindex(index).ffill().bfill() / 100.0
    daily = (1.0 + annual) ** (1.0 / cfg.backtest.periods_per_year) - 1.0
    return daily.rename("risk_free")


def load_market_weights(cfg: Config, prices: pd.DataFrame, data_root: Path,
                        refresh: bool = False) -> pd.DataFrame:
    """Reference portfolio used as the Black-Litterman prior.

    * ``method: shares_outstanding`` - cap_t = shares x price_t (today's share
      count is held fixed, an approximation documented in the README).
    * ``method: fixed`` - static weights given in the config (e.g. a global
      multi-asset market portfolio for ETFs, where "market cap" is ill-defined).
    * ``method: equal`` - 1/N prior.
    """
    spec = cfg.data.market_caps or {"method": "equal"}
    method = spec.get("method", "equal")
    tickers = list(prices.columns)

    if method == "equal":
        w = pd.DataFrame(1.0 / len(tickers), index=prices.index, columns=tickers)
    elif method == "fixed":
        raw = pd.Series(spec["weights"], dtype=float).reindex(tickers)
        if raw.isna().any():
            raise ValueError(f"Fixed market weights missing for {raw[raw.isna()].index.tolist()}")
        raw = raw / raw.sum()
        w = pd.DataFrame(np.tile(raw.values, (len(prices), 1)), index=prices.index,
                         columns=tickers)
    elif method == "shares_outstanding":
        path = _cache_dir(cfg, data_root) / "shares_outstanding.csv"
        if path.exists() and not refresh:
            shares = pd.read_csv(path, index_col=0).iloc[:, 0]
        else:
            shares = _download_shares_outstanding(tickers)
            shares.to_csv(path)
        caps = prices * shares.reindex(tickers)
        w = caps.div(caps.sum(axis=1), axis=0)
    else:
        raise ValueError(f"Unknown market_caps method: {method}")
    return w


def mask_stale_prices(prices: pd.DataFrame, min_run: int = STALE_RUN_DAYS) -> pd.DataFrame:
    """Set runs of >= ``min_run`` identical consecutive closes to NaN.

    Vendors sometimes back-fill a constant price before a listing's data really
    starts (e.g. NESTLEIND.NS throughout 2009). Real liquid stocks essentially
    never close unchanged for a week, so such runs are treated as missing.
    """
    out = prices.copy()
    for col in out:
        s = out[col]
        valid = s.dropna()
        block = (valid.diff() != 0).cumsum()
        run_len = valid.groupby(block).transform("size")
        stale = run_len[run_len >= min_run].index
        if len(stale):
            log.warning("%s: %d stale prices masked (%s -> %s)", col, len(stale),
                        stale[0].date(), stale[-1].date())
            out.loc[stale, col] = np.nan
    return out


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Mask stale quotes, fill short gaps and start where every asset has history."""
    prices = mask_stale_prices(prices.sort_index()).dropna(how="all")
    prices = prices.ffill(limit=MAX_FFILL_DAYS)
    first_full = prices.dropna().index.min()
    prices = prices.loc[first_full:].dropna()
    return prices


def load_market_data(cfg: Config, data_root: str | Path = "data",
                     refresh: bool = False) -> MarketData:
    data_root = Path(data_root)
    prices = clean_prices(load_prices(cfg, data_root, refresh))
    returns = prices.pct_change().iloc[1:]

    extreme = returns.abs().stack()
    extreme = extreme[extreme > EXTREME_DAILY_MOVE]
    for (date, ticker), move in extreme.items():
        log.warning("Large daily move %s %s: %.1f%% (kept; verify if suspicious)",
                    ticker, date.date(), 100 * move)

    rf = load_risk_free(cfg, returns.index, data_root, refresh)
    mkt = load_market_weights(cfg, prices, data_root, refresh).reindex(returns.index)
    log.info("Data: %d assets, %s -> %s (%d days)", returns.shape[1],
             returns.index[0].date(), returns.index[-1].date(), len(returns))
    return MarketData(prices=prices.loc[returns.index], returns=returns, risk_free=rf,
                      market_weights=mkt)
