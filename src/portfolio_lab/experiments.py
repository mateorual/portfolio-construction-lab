"""The research pipeline: one call reproduces every table and figure for a config.

Outputs (``results/<config name>/``)::

    metrics.csv / .md              performance + portfolio-structure statistics
    bootstrap_sharpe.csv           Sharpe CIs and paired differences vs 1/N
    cost_sensitivity.csv           Sharpe at several cost levels
    covariance_ablation.csv        Sharpe / vol / turnover by covariance estimator
    weight_stability.csv           dispersion of weights under resampled inputs
    bl_anatomy_<date>.csv          sample mean vs prior vs posterior at the last rebalance
    weights/<strategy>.csv         target weights at every rebalance
    returns.csv                    daily net returns of every strategy
    figures/*.png                  all charts
    REPORT.md                      auto-generated summary that embeds the above
    summary.json                   machine-readable headline numbers
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import optimizers as opt
from . import plotting as P
from .backtest import compute_estimates, rebalance_dates, run_backtests
from .config import Config
from .data import load_market_data
from .estimators import historical_mean
from .metrics import block_bootstrap_indices
from .strategies import BlackLitterman, Estimates, build_strategies, make_estimates

log = logging.getLogger(__name__)

BENCHMARK = P.BENCHMARK_ID
PCT_COLS = {"CAGR", "Volatility", "Max drawdown", "Turnover (1-way, p.a.)", "Avg max weight"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def markdown_table(df: pd.DataFrame, pct_cols: set[str] | None = None, decimals: int = 2,
                   index_label: str = "") -> str:
    pct_cols = pct_cols or set()

    def fmt(col, v):
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if col in pct_cols:
            return f"{v:.1%}"
        return f"{v:.{decimals}f}"

    header = "| " + " | ".join([index_label] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "|".join([":---"] + ["---:"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join([str(idx)] + [fmt(c, df.loc[idx, c]) for c in df.columns]) + " |"
            for idx in df.index]
    return "\n".join([header, sep, *rows])


def _relabel(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    return df.rename(index=labels)


# --------------------------------------------------------------------------- #
# Final-date analyses
# --------------------------------------------------------------------------- #
def weight_resampling(last: Estimates, strategies, cfg: Config, market_weights: pd.Series
                      ) -> dict[str, pd.DataFrame]:
    """Re-estimate inputs on block-bootstrap resamples of the final window.

    Quantifies *estimation-error sensitivity*: how much would the portfolio
    change if history had been a slightly different draw from the same process?
    """
    ex = cfg.experiments
    window = last.returns
    idx = block_bootstrap_indices(len(window), ex.bootstrap_block, ex.resampling_draws,
                                  ex.seed + 1)
    draws: dict[str, list[pd.Series]] = {s.id: [] for s in strategies}
    for k in range(ex.resampling_draws):
        w_k = pd.DataFrame(window.values[idx[k]], index=window.index, columns=window.columns)
        e_k = make_estimates(w_k, market_weights, cfg)
        for s in strategies:
            try:
                draws[s.id].append(s.target_weights(e_k))
            except opt.OptimizationError:
                continue
    return {sid: pd.DataFrame(v) for sid, v in draws.items()}


def frontier_table(e: Estimates, cfg: Config, n_points: int = 40) -> pd.DataFrame:
    lo, hi = cfg.constraints.min_weight, cfg.constraints.max_weight
    S = e.cov.values
    gmv = opt.min_variance(e.cov, lo, hi)
    r_lo = float(e.mu_hist @ gmv)
    r_hi = opt.max_return(e.mu_hist, lo, hi)
    rows = []
    for r in np.linspace(r_lo, r_hi - 1e-6 * abs(r_hi), n_points):
        w = opt.frontier_portfolio(e.mu_hist, e.cov, r, lo, hi).values
        rows.append({"ret": float(e.mu_hist.values @ w), "vol": float(np.sqrt(w @ S @ w))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(cfg: Config, data_root: str | Path = "data", results_root: str | Path = "results",
        refresh: bool = False) -> dict:
    t0 = time.time()
    P.apply_style()
    out = Path(results_root) / cfg.name
    figs = out / "figures"
    (out / "weights").mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    data = load_market_data(cfg, data_root, refresh)
    strategies = build_strategies(cfg)
    labels = {s.id: s.spec.label for s in strategies}
    ex = cfg.experiments
    periods = cfg.backtest.periods_per_year

    # ---- 1. main backtest ------------------------------------------------- #
    dates = rebalance_dates(data.returns.index, cfg.backtest.lookback_days, cfg.backtest.rebalance)
    log.info("[%s] %d rebalances %s -> %s", cfg.name, len(dates), dates[0].date(),
             dates[-1].date())
    estimates = compute_estimates(data, cfg, dates)
    covs = {d: e.cov for d, e in estimates.items()}
    results = run_backtests(data, strategies, cfg, estimates=estimates)
    net = pd.DataFrame({sid: r.returns for sid, r in results.items()})
    rf = data.risk_free.reindex(net.index).fillna(0.0)
    log.info("[%s] main backtest done (%.0fs)", cfg.name, time.time() - t0)

    # ---- 2. metrics ------------------------------------------------------- #
    rows = {}
    for sid, r in results.items():
        rows[sid] = {**M.performance_summary(r.returns, rf, periods),
                     **M.structure_diagnostics(r.target_weights, r.traded, covs, periods),
                     "Solver failures": float(len(r.failures))}
    metrics = pd.DataFrame(rows).T

    # ---- 3. inference ----------------------------------------------------- #
    boot = M.bootstrap_sharpe(net, rf, BENCHMARK, ex.bootstrap_block, ex.bootstrap_samples,
                              ex.seed, periods)

    # ---- 4. transaction-cost sensitivity ----------------------------------- #
    cost = pd.DataFrame({
        bps: {sid: M.sharpe_ratio(r.net_returns(bps), rf, periods) for sid, r in results.items()}
        for bps in ex.cost_sensitivity_bps
    }).T.rename_axis("cost_bps")

    # ---- 5. covariance-estimator ablation ---------------------------------- #
    ablation = {}
    for method in ex.covariance_ablation:
        if method == cfg.estimation.covariance:
            res_m = results
        else:
            est_m = compute_estimates(data, cfg, dates, method)
            res_m = run_backtests(data, strategies, cfg, estimates=est_m)
        for sid, r in res_m.items():
            ablation[(method, sid)] = {
                "Sharpe": M.sharpe_ratio(r.returns, rf, periods),
                "Volatility": float(r.returns.std() * np.sqrt(periods)),
                "Turnover (1-way, p.a.)": float(0.5 * r.traded.sum() / (len(r.traded) / periods)),
            }
        log.info("[%s] covariance ablation: %s done (%.0fs)", cfg.name, method, time.time() - t0)
    ablation = pd.DataFrame(ablation).T.rename_axis(["covariance", "strategy"])

    # ---- 6. structure through time ----------------------------------------- #
    group_w = {sid: M.group_shares(r.target_weights, cfg.groups, cfg.group_order)
               for sid, r in results.items()}
    risk_paths = {sid: M.risk_share_path(r.target_weights, covs) for sid, r in results.items()}
    capital = pd.DataFrame({sid: g.mean() for sid, g in group_w.items()}).T
    risk = pd.DataFrame({sid: M.group_shares(rp, cfg.groups, cfg.group_order).mean()
                         for sid, rp in risk_paths.items()}).T

    # ---- 7. final-date analyses -------------------------------------------- #
    last_date = dates[-1]
    last = estimates[last_date]
    mkt_last = data.market_weights.loc[last_date]
    bl_strategy = next((s for s in strategies if isinstance(s, BlackLitterman)), None)
    bl_table = None
    if bl_strategy is not None:
        post = bl_strategy.posterior(last)
        # BL works in excess returns; put the sample mean on the same footing
        rf_window = float(data.risk_free.reindex(last.returns.index).mean() * periods)
        bl_table = pd.DataFrame({
            "Sample mean": historical_mean(last.returns, periods) - rf_window,
            "Equilibrium prior": post.prior,
            "Momentum view": post.views,
            "Posterior (BL)": post.posterior,
        })
    resampled = weight_resampling(last, strategies, cfg, mkt_last)
    stability = pd.DataFrame({
        sid: {
            "Avg weight std": float(d.std().mean()),
            "Avg distance to median (1-way)": float(
                (d - d.median()).abs().sum(axis=1).mean() / 2),
            "Avg max weight": float(d.max(axis=1).mean()),
        } for sid, d in resampled.items() if len(d)
    }).T
    log.info("[%s] resampling done (%.0fs)", cfg.name, time.time() - t0)

    frontier = frontier_table(last, cfg)
    asset_pts = pd.DataFrame({"vol": np.sqrt(np.diag(last.cov.values)),
                              "ret": last.mu_hist.values}, index=last.cov.index)
    strat_pts = pd.DataFrame({
        sid: {"vol": float(np.sqrt(w @ last.cov.values @ w)), "ret": float(last.mu_hist.values @ w)}
        for sid, w in ((sid, r.target_weights.loc[last_date].values) for sid, r in results.items())
    }).T

    # ---- 8. figures -------------------------------------------------------- #
    title = cfg.title
    colors = P.strategy_colors(list(results))
    short = {a.ticker: a.ticker.replace(".NS", "") for a in cfg.assets}
    P.wealth_curves(net, labels, figs / "wealth.png", f"{title}: cumulative wealth", cfg.currency)
    P.drawdowns(net, labels, figs / "drawdowns.png", f"{title}: drawdowns")
    P.sharpe_intervals(boot, labels, figs / "sharpe_ci.png", f"{title}: Sharpe ratios")
    P.cost_sensitivity(cost, labels, figs / "cost_sensitivity.png",
                       f"{title}: sensitivity to trading costs", cfg.backtest.transaction_cost_bps)
    P.weights_by_group(group_w, labels, figs / "weights.png",
                       f"{title}: allocation by asset group through time", cfg.group_order)
    P.capital_vs_risk(capital, risk, labels, figs / "capital_vs_risk.png",
                      f"{title}: where the capital goes vs. where the risk comes from")
    P.weight_stability(resampled, labels, figs / "weight_stability.png",
                       f"{title}: sensitivity of weights to estimation error", short)
    if bl_table is not None:
        P.bl_anatomy(bl_table, figs / "bl_anatomy.png",
                     f"{title}: Black-Litterman expected returns ({last_date.date()})", short,
                     {"sample": colors.get("mean_variance", P.SLOTS[1]),
                      "posterior": colors.get(bl_strategy.id, P.SLOTS[2])})
    P.efficient_frontier(frontier, asset_pts, strat_pts, labels, figs / "efficient_frontier.png",
                         f"{title}: strategies on the ex-ante frontier ({last_date.date()})")

    # ---- 9. tables --------------------------------------------------------- #
    metrics.to_csv(out / "metrics.csv")
    boot.to_csv(out / "bootstrap_sharpe.csv")
    cost.to_csv(out / "cost_sensitivity.csv")
    ablation.to_csv(out / "covariance_ablation.csv")
    stability.to_csv(out / "weight_stability.csv")
    net.to_csv(out / "returns.csv", float_format="%.8f")
    if bl_table is not None:
        bl_table.to_csv(out / f"bl_anatomy_{last_date.date()}.csv")
    for sid, r in results.items():
        r.target_weights.to_csv(out / "weights" / f"{sid}.csv", float_format="%.6f")

    summary = {
        "config": cfg.name,
        "period": [str(net.index[0].date()), str(net.index[-1].date())],
        "n_rebalances": len(dates),
        "transaction_cost_bps": cfg.backtest.transaction_cost_bps,
        "covariance": cfg.estimation.covariance,
        "metrics": metrics.round(6).to_dict(orient="index"),
        "bootstrap": boot.round(6).to_dict(orient="index"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_report(cfg, out, labels, metrics, boot, cost, ablation, stability, net, len(dates))
    log.info("[%s] finished in %.0fs -> %s", cfg.name, time.time() - t0, out)
    return summary


def _write_report(cfg: Config, out: Path, labels: dict[str, str], metrics: pd.DataFrame,
                  boot: pd.DataFrame, cost: pd.DataFrame, ablation: pd.DataFrame,
                  stability: pd.DataFrame, net: pd.DataFrame, n_rebal: int) -> None:
    perf_cols = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar"]
    struct_cols = ["Turnover (1-way, p.a.)", "Effective N (weights)", "Effective N (risk)",
                   "Avg max weight", "Diversification ratio"]
    boot_view = boot[["Sharpe", "Sharpe CI low", "Sharpe CI high", "dSharpe vs benchmark",
                      "dSharpe CI low", "dSharpe CI high", "p (dSharpe <= 0)"]]
    abl = ablation["Sharpe"].unstack(0).reindex(list(labels))[list(cfg.experiments.covariance_ablation)]
    abl_turn = ablation["Turnover (1-way, p.a.)"].unstack(0).reindex(list(labels))[
        list(cfg.experiments.covariance_ablation)]
    cost_view = cost.T.reindex(list(labels))
    cost_view.columns = [f"{c:g} bps" for c in cost_view.columns]

    lines = [
        f"# {cfg.title} - results",
        "",
        f"_Auto-generated by `portfolio-lab run {cfg.source_path.as_posix() if cfg.source_path else ''}`._",
        "",
        f"* Out-of-sample period: **{net.index[0].date()} -> {net.index[-1].date()}** "
        f"({len(net):,} trading days, {n_rebal} monthly rebalances)",
        f"* Estimation window: {cfg.backtest.lookback_days} trading days; "
        f"covariance: `{cfg.estimation.covariance}`",
        f"* Transaction costs: {cfg.backtest.transaction_cost_bps:g} bps per unit traded; "
        f"currency: {cfg.currency}",
        "",
        "![Wealth](figures/wealth.png)",
        "",
        "## Performance (net of costs)",
        "",
        markdown_table(_relabel(metrics[perf_cols], labels), PCT_COLS),
        "",
        "## Portfolio structure (averages over rebalances)",
        "",
        markdown_table(_relabel(metrics[struct_cols], labels), PCT_COLS),
        "",
        "![Capital vs risk](figures/capital_vs_risk.png)",
        "",
        "![Weights](figures/weights.png)",
        "",
        "## Is the difference real? Block-bootstrap Sharpe inference vs 1/N",
        "",
        markdown_table(_relabel(boot_view, labels), decimals=2),
        "",
        "![Sharpe CI](figures/sharpe_ci.png)",
        "",
        "## Covariance estimator ablation",
        "",
        "Sharpe ratio:",
        "",
        markdown_table(_relabel(abl, labels)),
        "",
        "One-way annual turnover:",
        "",
        markdown_table(_relabel(abl_turn, labels), set(abl_turn.columns)),
        "",
        "## Transaction-cost sensitivity (Sharpe)",
        "",
        markdown_table(_relabel(cost_view, labels)),
        "",
        "![Costs](figures/cost_sensitivity.png)",
        "",
        "## Estimation-error sensitivity (final window, resampled)",
        "",
        markdown_table(_relabel(stability, labels), {"Avg distance to median (1-way)",
                                                     "Avg max weight", "Avg weight std"}),
        "",
        "![Stability](figures/weight_stability.png)",
        "",
        "## Black-Litterman anatomy and the efficient frontier",
        "",
        "![BL](figures/bl_anatomy.png)",
        "",
        "![Frontier](figures/efficient_frontier.png)",
        "",
        "![Drawdowns](figures/drawdowns.png)",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
