"""Figures. One visual system for every chart in the project.

* Strategies keep the same colour in every figure; the 1/N benchmark is a
  neutral grey so the eye goes to the optimised portfolios.
* Categorical hues come in a fixed, colour-vision-deficiency-validated order
  (never cycled); asset groups use the same slot order.
* Hairline solid grids, 2px lines, no top/right spines, text in ink colours
  (never in series colours), a legend whenever there are >= 2 series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---- design tokens (light surface) --------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BENCHMARK = "#898781"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK_TEXT_FILLS = {"#eda100"}  # fills light enough to need near-black labels

BENCHMARK_ID = "equal_weight"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def strategy_colors(strategy_ids: list[str]) -> dict[str, str]:
    colors, k = {}, 0
    for sid in strategy_ids:
        if sid == BENCHMARK_ID:
            colors[sid] = BENCHMARK
        else:
            colors[sid] = SLOTS[k]
            k += 1
    return colors


def group_colors(groups: list[str]) -> dict[str, str]:
    if len(groups) > len(SLOTS):
        raise ValueError("More than 8 groups: fold small groups into 'Other'")
    return {g: SLOTS[i] for i, g in enumerate(groups)}


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def _finish(fig, path: Path, title: str, subtitle: str | None = None,
            legend: tuple[list, list[str]] | None = None) -> Path:
    """Header block (title, subtitle, optional legend row) above a tight plot area.

    Heights are in inches so spacing is identical across figure sizes.
    """
    H = fig.get_figheight()
    title_h, sub_h, leg_h = 0.34, 0.26, 0.34
    header = 0.12 + title_h + (sub_h if subtitle else 0) + (leg_h if legend else 0)
    fig.tight_layout(rect=(0, 0, 1, 1 - header / H))
    y = 1 - 0.08 / H
    fig.text(0.01, y, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    y -= title_h / H
    if subtitle:
        fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
        y -= sub_h / H
    if legend:
        handles, labels = legend
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, y + 0.02 / H),
                   ncol=len(labels), handlelength=1.0, handleheight=1.0, columnspacing=1.6,
                   borderaxespad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _log_wealth_ticks(ax, lo: float, hi: float) -> None:
    nice = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
    ticks = [t for t in nice if lo * 0.95 <= t <= hi * 1.05]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))


def _swatches(colors: list[str]) -> list:
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]


# --------------------------------------------------------------------------- #
# Performance figures
# --------------------------------------------------------------------------- #
def wealth_curves(net: pd.DataFrame, labels: dict[str, str], path: Path, title: str,
                  currency: str) -> Path:
    colors = strategy_colors(list(net.columns))
    wealth = (1.0 + net).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for sid in net.columns:
        bench = sid == BENCHMARK_ID
        ax.plot(wealth.index, wealth[sid], color=colors[sid], label=labels[sid],
                zorder=1 if bench else 2, lw=1.6 if bench else 1.9)
    ax.set_yscale("log")
    _log_wealth_ticks(ax, float(wealth.min().min()), float(wealth.max().max()))
    ax.set_ylabel(f"Growth of 1 {currency} (log scale)")
    ax.margins(x=0.01)
    handles, lbls = ax.get_legend_handles_labels()
    return _finish(fig, path, title, "Net of transaction costs, monthly rebalancing, strictly "
                   "out-of-sample", legend=(handles, lbls))


def drawdowns(net: pd.DataFrame, labels: dict[str, str], path: Path, title: str) -> Path:
    colors = strategy_colors(list(net.columns))
    wealth = (1.0 + net).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    n = len(net.columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.3 * nrows), sharex=True, sharey=True)
    for ax, sid in zip(axes.flat, net.columns, strict=False):
        ax.fill_between(dd.index, dd[sid], 0, color=colors[sid], alpha=0.12, lw=0)
        ax.plot(dd.index, dd[sid], color=colors[sid], lw=1.4)
        worst = dd[sid].idxmin()
        ax.annotate(f"{dd[sid].min():.0%}", (worst, dd[sid].min()), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5, color=INK_2, va="center")
        ax.set_title(labels[sid])
        _pct_axis(ax)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    return _finish(fig, path, title, "Peak-to-trough decline; the label marks each strategy's "
                   "worst point")


def sharpe_intervals(boot: pd.DataFrame, labels: dict[str, str], path: Path, title: str) -> Path:
    colors = strategy_colors(list(boot.index))
    order = boot.sort_values("Sharpe").index
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 0.5 * len(order) + 0.9), sharey=True,
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    panels = [(ax, "Sharpe", "Sharpe CI low", "Sharpe CI high", "Sharpe ratio"),
              (ax2, "dSharpe vs benchmark", "dSharpe CI low", "dSharpe CI high",
               "Difference vs equal weight (paired)")]
    for a, point, lo, hi, name in panels:
        for y, sid in enumerate(order):
            r = boot.loc[sid]
            if np.isnan(r[lo]):  # the benchmark itself in the difference panel
                a.scatter(0, y, s=30, color=colors[sid], zorder=3)
                continue
            a.plot([r[lo], r[hi]], [y, y], color=colors[sid], lw=3, alpha=0.4,
                   solid_capstyle="round")
            a.scatter(r[point], y, s=70, color=colors[sid], edgecolor=SURFACE, linewidth=2,
                      zorder=3)
            a.annotate(f"{r[point]:+.2f}" if a is ax2 else f"{r[point]:.2f}", (r[hi], y),
                       xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                       color=INK_2)
        a.grid(axis="y", visible=False)
        a.axvline(0, color=AXIS, lw=1)
        a.set_title(name)
    ax.set_yticks(range(len(order)), [labels[s] for s in order])
    ax.set_xlabel("Annualised, excess of risk-free")
    ax2.set_xlabel("Sharpe difference")
    return _finish(fig, path, title, "Point estimates with 95% circular block-bootstrap intervals "
                   "(21-day blocks, 2,000 resamples). Pairing removes the common market noise, "
                   "so the right panel is the real test.")


def cost_sensitivity(table: pd.DataFrame, labels: dict[str, str], path: Path, title: str,
                     base_bps: float) -> Path:
    """``table``: index = bps, columns = strategy ids, values = Sharpe."""
    colors = strategy_colors(list(table.columns))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for sid in table.columns:
        ax.plot(table.index, table[sid], color=colors[sid], label=labels[sid], marker="o",
                ms=5, mec=SURFACE, mew=1.5)
    ax.axvline(base_bps, color=AXIS, lw=1)
    ax.annotate("base case", (base_bps, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points", fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel("Transaction cost (bps per unit traded)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _finish(fig, path, title, "High-turnover methods lose their edge fastest as trading "
                   "gets more expensive")


# --------------------------------------------------------------------------- #
# Portfolio-structure figures
# --------------------------------------------------------------------------- #
def weights_by_group(weights: dict[str, pd.DataFrame], labels: dict[str, str], path: Path,
                     title: str, group_order: list[str]) -> Path:
    gcol = group_colors(group_order)
    ids = list(weights)
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.5 * nrows), sharex=True, sharey=True)
    for ax, sid in zip(axes.flat, ids, strict=False):
        w = weights[sid].reindex(columns=group_order, fill_value=0.0)
        ax.stackplot(w.index, w.T.values, colors=[gcol[g] for g in group_order],
                     edgecolor=SURFACE, linewidth=0.5, step="post")
        ax.set_ylim(0, 1)
        ax.set_title(labels[sid])
        ax.grid(axis="x", visible=False)
        _pct_axis(ax)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    return _finish(fig, path, title, "Target weights at each monthly rebalance, aggregated by "
                   "asset group", legend=(_swatches([gcol[g] for g in group_order]), group_order))


def capital_vs_risk(capital: pd.DataFrame, risk: pd.DataFrame, labels: dict[str, str],
                    path: Path, title: str) -> Path:
    """Average group share of capital vs. of ex-ante risk, one bar per strategy."""
    groups = list(capital.columns)
    gcol = group_colors(groups)
    ids = list(capital.index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.48 * len(ids) + 0.9), sharey=True)
    for ax, data, name in [(axes[0], capital, "Share of capital"),
                           (axes[1], risk, "Share of ex-ante risk (Euler contributions)")]:
        left = np.zeros(len(ids))
        for g in groups:
            vals = data.loc[ids, g].values
            ax.barh(range(len(ids)), vals, left=left, height=0.62, color=gcol[g],
                    edgecolor=SURFACE, linewidth=1.5)
            for y, (lft, v) in enumerate(zip(left, vals, strict=True)):
                if v >= 0.09:
                    ink = INK if gcol[g] in DARK_TEXT_FILLS else "#ffffff"
                    ax.text(lft + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8,
                            color=ink)
            left += vals
        ax.set_xlim(0, 1)
        ax.set_title(name)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        _pct_axis(ax, "x")
    axes[0].set_yticks(range(len(ids)), [labels[s] for s in ids])
    return _finish(fig, path, title, "Averages over all rebalance dates. Equal capital is not "
                   "equal risk.", legend=(_swatches([gcol[g] for g in groups]), groups))


def weight_stability(draws: dict[str, pd.DataFrame], labels: dict[str, str], path: Path,
                     title: str, asset_labels: dict[str, str]) -> Path:
    """Spread of weights across bootstrap resamples of the final estimation window."""
    colors = strategy_colors(list(draws))
    ids = [s for s in draws if s != BENCHMARK_ID]
    assets = list(next(iter(draws.values())).columns)
    ncols = len(ids)
    fig, axes = plt.subplots(1, ncols, figsize=(2.3 * ncols + 1.2, 0.26 * len(assets) + 0.9),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(assets))[::-1]
    for ax, sid in zip(axes, ids, strict=False):
        d = draws[sid]
        lo, med, hi = d.quantile(0.05), d.median(), d.quantile(0.95)
        ax.hlines(y, lo.values, hi.values, color=colors[sid], lw=3.5, alpha=0.4)
        ax.scatter(med.values, y, color=colors[sid], s=24, edgecolor=SURFACE, linewidth=1,
                   zorder=3)
        ax.set_title(labels[sid], fontsize=9.5)
        ax.grid(axis="y", visible=False)
        _pct_axis(ax, "x")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
    axes[0].set_yticks(y, [asset_labels.get(a, a) for a in assets], fontsize=8.5)
    return _finish(fig, path, title, "Median weight (dot) and 5-95% range (bar) when the last "
                   "estimation window is block-bootstrap resampled")


def bl_anatomy(table: pd.DataFrame, path: Path, title: str, asset_labels: dict[str, str],
               colors: dict[str, str]) -> Path:
    """Sample mean vs. equilibrium prior vs. Black-Litterman posterior, per asset."""
    t = table.sort_values("Posterior (BL)")
    series = [("Sample mean", colors["sample"]), ("Equilibrium prior", BENCHMARK),
              ("Posterior (BL)", colors["posterior"])]
    y = np.arange(len(t))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(t) + 0.9))
    for k, (col, c) in enumerate(series):
        ax.barh(y + (1 - k) * h, t[col].values, height=h * 0.9, color=c, label=col)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [asset_labels.get(a, a) for a in t.index], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    _pct_axis(ax, "x")
    ax.set_xlabel("Expected annual excess return")
    handles, lbls = ax.get_legend_handles_labels()
    return _finish(fig, path, title, "The posterior stays anchored to the equilibrium prior and "
                   "tilts toward momentum; the trailing sample mean swings wildly",
                   legend=(handles, lbls))


def efficient_frontier(frontier: pd.DataFrame, assets: pd.DataFrame, points: pd.DataFrame,
                       labels: dict[str, str], path: Path, title: str) -> Path:
    """Ex-ante (in-sample) frontier at the final rebalance date with strategy positions."""
    colors = strategy_colors(list(points.index))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(frontier["vol"], frontier["ret"], color=INK_2, lw=1.6, label="Efficient frontier")
    ax.scatter(assets["vol"], assets["ret"], s=26, color=AXIS, edgecolor=SURFACE, zorder=2,
               label="Individual assets")
    for sid, r in points.iterrows():
        ax.scatter(r["vol"], r["ret"], s=90, color=colors[sid], edgecolor=SURFACE, linewidth=2,
                   zorder=4, label=labels[sid])
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(mticker.MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    _pct_axis(ax, "x")
    _pct_axis(ax, "y")
    # selective labels: place greedily, skip any that would collide with one already placed
    fig.canvas.draw()
    placed: list = []
    for a, r in assets.sort_values("ret", ascending=False).iterrows():
        txt = ax.annotate(a.replace(".NS", ""), (r["vol"], r["ret"]), xytext=(4, 2),
                          textcoords="offset points", fontsize=7.5, color=MUTED)
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.15)
        if any(bb.overlaps(p) for p in placed):
            txt.remove()
        else:
            placed.append(bb)
    ax.set_xlabel("Volatility (ex-ante)")
    ax.set_ylabel("Expected return (trailing sample mean)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _finish(fig, path, title, "In-sample view at the final rebalance: mean-variance sits on "
                   "the frontier by construction - which is exactly why it disappoints out of "
                   "sample")
