# Methodology

This document specifies every modelling choice in the project, so that each number
in the README can be traced to an equation and a line of code.

- [1. Backtest protocol](#1-backtest-protocol)
- [2. Estimators](#2-estimators)
- [3. Portfolio construction methods](#3-portfolio-construction-methods)
- [4. Evaluation](#4-evaluation)
- [5. Implementation pitfalls (and how this code avoids them)](#5-implementation-pitfalls-and-how-this-code-avoids-them)
- [References](#references)

---

## 1. Backtest protocol

| Item | Choice | Code |
|---|---|---|
| Rebalancing | Last trading day of each month | `backtest.rebalance_dates` |
| Information set | The trailing 252 daily returns **up to and including** the rebalance date | `backtest.compute_estimates` |
| Execution | At the close of the rebalance date; new weights earn returns from the next day | `backtest.simulate` |
| Between rebalances | Weights **drift** with prices: $w_{t+1} = w_t \odot (1+r_t) / (1 + w_t^\top r_t)$ | `backtest.simulate` |
| Trading cost | $c \cdot \lVert w^{target} - w^{drifted} \rVert_1$, charged on the rebalance day | `BacktestResult.net_returns` |
| Initial buy-in | Not charged (identical for every strategy) | |
| Solver failure | Keep the previous target and record the date | `backtest.compute_targets` |

All strategies consume the **same** `Estimates` object at each date, so performance
differences come only from the construction rule.

Two tests enforce the absence of look-ahead: one asserts no estimate ever contains a
date after its rebalance date; the other rewrites all returns after a cut-off date and
checks that every decision before the cut-off is bit-for-bit unchanged
(`tests/test_backtest.py`).

## 2. Estimators

All estimators return **annualised** quantities (×252).

**Sample covariance** $\hat\Sigma = \frac{1}{T-1}\sum_t (r_t-\bar r)(r_t-\bar r)^\top$.

**Ledoit–Wolf shrinkage** (default) $\hat\Sigma_{LW} = (1-\alpha)\hat\Sigma + \alpha\,\bar\sigma^2 I$
with the analytically optimal intensity $\alpha$ (Ledoit & Wolf, 2004). Shrinkage pulls the
extreme eigenvalues of $\hat\Sigma$ toward their mean. Those extreme eigenvalues dominate
$\hat\Sigma^{-1}$ and therefore every optimiser.

**EWMA** with half-life $h$ = 63 days: weights $\propto \lambda^{T-t}$, $\lambda = 0.5^{1/h}$.

**Expected returns**: the annualised sample mean $\hat\mu = 252\,\bar r$. Its standard error
is $\sigma/\sqrt{T}$ (for $\sigma=20\%$ and one year of data, ±20% per year), which is
the root cause of mean–variance instability.

**Momentum signal** (Black–Litterman views): the compounded return from $t-252$ to $t-21$,
skipping the latest month to avoid short-term reversal (Jegadeesh & Titman, 1993).

## 3. Portfolio construction methods

All optimised portfolios are long-only and fully invested: $w \ge 0$, $\mathbf 1^\top w = 1$
(plus an optional cap $w \le \bar w$, set to 1 by default so every method shows its
natural behaviour).

### 3.1 Equal weight (benchmark)
$w_i = 1/N$. It needs no estimation and therefore has zero estimation error, which is
why it is hard to beat (DeMiguel, Garlappi & Uppal, 2009).

### 3.2 Global minimum variance
$$\min_w\ w^\top\Sigma w.$$
Uses only $\Sigma$, so it is immune to errors in $\mu$.

### 3.3 Mean–variance (sample mean)
$$\max_w\ \hat\mu^\top w - \tfrac{\delta}{2}\,w^\top\Sigma w, \qquad \delta = 2.5.$$
This is the textbook Markowitz portfolio with the plug-in estimate $\hat\mu$. It is
included deliberately as the "estimation-error maximiser" (Michaud, 1989).

### 3.4 Black–Litterman with momentum views
1. **Prior (reverse optimisation).** The excess returns that make the reference
   portfolio $w_m$ optimal: $\pi = \delta\Sigma w_m$.
   *Global multi-asset:* an illustrative global market portfolio (≈50% equity, 40% bonds,
   10% real assets). *Indian equities:* market caps, approximated as today's
   shares outstanding × price.
2. **Views.** One absolute view per asset from Grinold's rule
   $Q_i = \pi_i + \mathrm{IC}\cdot\sigma_i\cdot z_i$, where $z_i$ is the cross-sectional
   z-score of momentum (clipped at ±3) and IC = 0.10. A flat signal reproduces the prior.
3. **View uncertainty** (He & Litterman, 1999): $\Omega = \mathrm{diag}(P\,\tau\Sigma\,P^\top)$, $\tau = 0.05$.
4. **Posterior.**
$$\mu_{BL} = \big[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\big]^{-1}\big[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\big],
\qquad \Sigma_{BL} = \Sigma + \big[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\big]^{-1}.$$
5. **Portfolio.** Mean–variance on $(\mu_{BL}, \Sigma_{BL})$ with the same $\delta$.
   With no views and no constraints this returns $w_m$ exactly (unit-tested).

### 3.5 Risk parity (equal risk contribution)
Risk contributions come from the Euler decomposition of volatility,
$RC_i = w_i(\Sigma w)_i/\sqrt{w^\top\Sigma w}$, with $\sum_i RC_i = \sigma_p$.
ERC requires $RC_i = \sigma_p/N$ for all $i$. Following Spinu (2013), solve the
**strictly convex** problem
$$\min_{y>0}\ \tfrac12 y^\top\Sigma y - \sum_i b_i\log y_i, \qquad w = y/\mathbf 1^\top y,$$
whose first-order condition $y_i(\Sigma y)_i = b_i$ is exactly the risk-budget condition.
It is solved by damped Newton iterations (gradient $\Sigma y - b/y$, Hessian
$\Sigma + \mathrm{diag}(b/y^2)$), which converge quadratically to machine precision.
cvxpy serves as a fallback.

### 3.6 Relaxed risk parity
Gambeta & Kwon (2020) relax ERC so that it can reach a return target $R$:
$$\begin{aligned}
\min_{x,\psi,\gamma,\rho}\ & \psi - \gamma\\
\text{s.t.}\ & \zeta = \Sigma x,\quad x_i\zeta_i \ge n\,b_i\,\gamma^2 \quad (\text{rotated cones})\\
& \tfrac1n x^\top\Sigma x + \rho^2 \le \psi^2,\qquad \lambda\,x^\top\Theta x \le \rho^2\\
& \mu^\top x \ge R,\qquad \mathbf 1^\top x = 1,\ x \ge 0 .
\end{aligned}$$
With equal budgets $\gamma \le \sigma/\sqrt n \le \psi$, so the objective is zero exactly
at ERC. The target is $R = r_{ERC} + \kappa\,(r_{max} - r_{ERC})$ with $\kappa = 0.25$.
Here $\mu = \hat\mu$, the same noisy sample mean as in §3.3, $\lambda = 0$, and
$\Theta = \mathrm{diag}(\Sigma)$.

*Feasibility note.* The cone constraints force every marginal risk $(\Sigma x)_i$ to be
non-negative. With negatively correlated assets (e.g. Treasuries vs equities), a high
target can therefore be unattainable. In that case the code bisects to the largest
feasible $\kappa$ instead of failing.

## 4. Evaluation

**Performance** (daily net returns, $r^f$ = 13-week T-bill for USD, 6.5% flat for INR):

| Metric | Definition |
|---|---|
| CAGR | $\big(\prod_t(1+r_t)\big)^{252/T} - 1$ |
| Volatility | $\sqrt{252}\,\mathrm{sd}(r_t)$ |
| Sharpe | $\sqrt{252}\,\overline{(r-r^f)}/\mathrm{sd}(r-r^f)$, arithmetic, excess of cash |
| Sortino | $252\,\overline{(r-r^f)} / \big(\sqrt{252}\sqrt{\overline{\min(r-r^f,0)^2}}\big)$ |
| Max drawdown | $\min_t W_t/\max_{s\le t}W_s - 1$ |
| Calmar | CAGR / \|max drawdown\| |

**Portfolio structure** (averaged over rebalance dates, using the covariance available
*at* that date):

| Metric | Definition |
|---|---|
| Turnover (one-way, p.a.) | $\tfrac12\sum \lVert\Delta w\rVert_1$ / years |
| Effective N (weights) | $1/\sum_i w_i^2$ |
| Effective N (risk) | $1/\sum_i (RC_i/\sigma_p)^2$, equal to $N$ for ERC by construction |
| Diversification ratio | $w^\top\sigma / \sqrt{w^\top\Sigma w}$ (Choueifaty & Coignard, 2008) |

**Inference.** Sharpe ratios estimated over ~18 years still carry standard errors of
roughly ±0.25. Confidence intervals come from a **circular block bootstrap**
(Politis & Romano, 1992; 21-day blocks, 2,000 resamples). Blocks keep volatility
clustering intact. Every strategy is resampled with the *same* indices, so
$\Delta SR = SR_k - SR_{1/N}$ keeps the strong positive correlation between strategies.
That makes the paired test far more powerful than comparing two separate intervals.
The reported $p$ is the one-sided share of resamples with $\Delta SR \le 0$.

**Robustness.**
1. *Covariance ablation*: the whole backtest re-run with sample / Ledoit–Wolf / EWMA.
2. *Cost sensitivity*: Sharpe at 0–50 bps, cheap to compute because gross returns and
   traded notional are stored separately.
3. *Estimation-error sensitivity*: the final 252-day window is block-bootstrapped 300
   times, and every input (Σ, μ, momentum) and every portfolio is re-estimated. The
   spread of the resulting weights measures how much each method amplifies sampling noise.

## 5. Implementation pitfalls (and how this code avoids them)

| Pitfall | Symptom | Fix here |
|---|---|---|
| Feeding the Black–Litterman posterior into a **minimum-variance** optimiser | Views have (almost) no effect: min-variance never reads $\mu$; only $\Sigma_{BL}$ changes slightly | Posterior fed to mean–variance (§3.4) |
| Risk parity as $\min \sum_i (RC_i - \overline{RC})^2$ with SLSQP on **daily** covariances | Objective ≈ 1e-6 is below the default tolerance, so the optimiser returns the equal-weight starting point unchanged | Spinu's convex formulation with scale normalisation; a regression test checks daily and annual units give identical weights |
| "Relaxed" risk parity as an ad-hoc mix $\lambda\,\text{risk error} - (1-\lambda)\,\mu^\top w$ | Mixes units (variance² vs daily return); the trade-off depends on the data frequency | Gambeta–Kwon SOCP with an explicit return target |
| Evaluating weights only in-sample (stability, HHI) | Says nothing about realised risk/return | Monthly walk-forward backtest with costs, bootstrap inference |
| Ignoring weight drift between rebalances | Turnover and costs understated | Explicit drift; turnover measured against drifted weights |
| Z-scoring a constant signal | Floating-point noise (σ ≈ 1e-17) becomes ±1 z-scores, i.e. fake views | Dispersion guard; unit-tested |
| Vendor back-filled prices | A year of constant prices, then a +76% "return" (NESTLEIND.NS, 2009) | Runs of ≥5 identical closes are masked; the sample starts after them |
| Static, hand-typed market caps | Prior frozen at one date | Caps evolve with prices (shares × price), or explicit asset-class weights for ETFs |

## References

- Black, F. & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5).
- Choueifaty, Y. & Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management*, 35(1).
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*, 22(5).
- Doeswijk, R., Lam, T. & Swinkels, L. (2014). The global multi-asset market portfolio, 1959–2012. *Financial Analysts Journal*, 70(2).
- Gambeta, V. & Kwon, R. (2020). Risk return trade-off in relaxed risk parity portfolio optimization. *Journal of Risk and Financial Management*, 13(10).
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- He, G. & Litterman, R. (1999). The intuition behind Black–Litterman model portfolios. Goldman Sachs.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1).
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2).
- Maillard, S., Roncalli, T. & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4).
- Michaud, R. (1989). The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal*, 45(1).
- Politis, D. & Romano, J. (1992). A circular block-resampling procedure for stationary data. In *Exploring the Limits of Bootstrap*. Wiley.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.
