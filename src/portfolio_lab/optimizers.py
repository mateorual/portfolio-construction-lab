r"""Convex portfolio optimizers (long-only, fully invested).

Every problem is posed so that an off-the-shelf conic solver returns the *global*
optimum - no starting points, no tolerances that silently stop at the initial
guess. In particular:

* **Risk parity (ERC)** uses Spinu's (2013) convex reformulation

  .. math:: \min_{y>0}\ \tfrac12 y^\top\Sigma y - \sum_i b_i \log y_i,
            \qquad w = y / \mathbf 1^\top y

  whose first-order condition :math:`y_i(\Sigma y)_i = b_i` *is* the risk-budget
  condition. The popular alternative - minimising the squared dispersion of risk
  contributions with SLSQP - is non-convex and badly scaled: with daily
  covariances its objective is ~1e-6, below SLSQP's default tolerance, so it can
  return the equal-weight starting point unchanged.

* **Relaxed risk parity** follows Gambeta & Kwon (2020), a second-order cone
  programme that recovers ERC when its return constraint is slack and trades
  away risk-contribution equality for expected return as the target rises.
"""

from __future__ import annotations

import logging
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SOLVER_PREFERENCE = ("CLARABEL", "ECOS", "SCS")
_OK = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


class OptimizationError(RuntimeError):
    """Raised when no solver returns an optimal solution."""


def _solve(problem: cp.Problem) -> None:
    installed = set(cp.installed_solvers())
    last_status = None
    for solver in _SOLVER_PREFERENCE:
        if solver not in installed:
            continue
        try:
            with warnings.catch_warnings():
                # "Solution may be inaccurate" - OPTIMAL_INACCURATE is accepted below
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(solver=solver)
        except cp.SolverError:
            continue
        last_status = problem.status
        if problem.status in _OK:
            return
    raise OptimizationError(f"No solver converged (last status: {last_status})")


def _finalize(w: np.ndarray, index: pd.Index, min_weight: float = 0.0) -> pd.Series:
    """Remove solver dust (|w| ~ 1e-9) and renormalise to exactly 1."""
    w = np.asarray(w, dtype=float).ravel()
    if min_weight >= 0:
        w = np.where(w < 1e-8, 0.0, w)
    return pd.Series(w / w.sum(), index=index)


def _as_arrays(cov: pd.DataFrame, mu: pd.Series | None = None):
    S = cov.values
    S = (S + S.T) / 2.0
    m = None if mu is None else mu.reindex(cov.index).values
    return S, m


def _box(w: cp.Variable, min_weight: float, max_weight: float) -> list:
    return [cp.sum(w) == 1, w >= min_weight, w <= max_weight]


# --------------------------------------------------------------------------- #
# Classical Markowitz family
# --------------------------------------------------------------------------- #
def min_variance(cov: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    """Global minimum-variance portfolio: min w'Sigma w."""
    S, _ = _as_arrays(cov)
    w = cp.Variable(len(S))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))), _box(w, min_weight, max_weight))
    _solve(prob)
    return _finalize(w.value, cov.index, min_weight)


def mean_variance(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float,
                  min_weight: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    """Quadratic-utility Markowitz portfolio: max mu'w - (delta/2) w'Sigma w."""
    S, m = _as_arrays(cov, mu)
    w = cp.Variable(len(S))
    utility = m @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(S))
    prob = cp.Problem(cp.Maximize(utility), _box(w, min_weight, max_weight))
    _solve(prob)
    return _finalize(w.value, cov.index, min_weight)


def frontier_portfolio(mu: pd.Series, cov: pd.DataFrame, target_return: float,
                       min_weight: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    """Minimum-variance portfolio with expected return >= ``target_return`` (one frontier point)."""
    S, m = _as_arrays(cov, mu)
    w = cp.Variable(len(S))
    cons = _box(w, min_weight, max_weight) + [m @ w >= target_return]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))), cons)
    _solve(prob)
    return _finalize(w.value, cov.index, min_weight)


def max_return(mu: pd.Series, min_weight: float = 0.0, max_weight: float = 1.0) -> float:
    """Highest expected return attainable under the weight box (an LP)."""
    w = cp.Variable(len(mu))
    prob = cp.Problem(cp.Maximize(mu.values @ w), _box(w, min_weight, max_weight))
    _solve(prob)
    return float(prob.value)


# --------------------------------------------------------------------------- #
# Risk budgeting family
# --------------------------------------------------------------------------- #
def risk_contributions(weights: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray,
                       relative: bool = True) -> np.ndarray:
    r"""Euler decomposition of volatility: :math:`RC_i = w_i(\Sigma w)_i / \sigma_p`.

    The contributions sum to portfolio volatility; with ``relative=True`` they
    are returned as shares that sum to one.
    """
    w = np.asarray(weights, dtype=float)
    S = np.asarray(cov, dtype=float)
    marginal = S @ w
    sigma = float(np.sqrt(w @ marginal))
    rc = w * marginal / sigma
    return rc / sigma if relative else rc


def _spinu_newton(S: np.ndarray, b: np.ndarray, tol: float = 1e-12, max_iter: int = 100
                  ) -> np.ndarray | None:
    r"""Damped Newton on :math:`f(y)=\tfrac12 y^\top S y - b^\top\log y`.

    :math:`\nabla f = Sy - b/y`, :math:`\nabla^2 f = S + \mathrm{diag}(b/y^2)` (positive
    definite), so Newton converges quadratically; the step is damped to keep y > 0.
    Returns None if it fails to converge (caller falls back to a conic solver).
    """
    y = 1.0 / np.sqrt(np.diag(S))
    y *= np.sqrt(b.sum() / (y @ S @ y))  # start on the right scale
    for _ in range(max_iter):
        g = S @ y - b / y
        H = S + np.diag(b / y**2)
        step = np.linalg.solve(H, g)
        decrement = float(g @ step)  # squared Newton decrement
        if decrement / 2 < tol:
            return y
        t = 1.0
        while np.any(y - t * step <= 0):  # stay inside the domain
            t *= 0.5
        f = 0.5 * y @ S @ y - b @ np.log(y)
        while True:  # Armijo backtracking
            y_new = y - t * step
            if 0.5 * y_new @ S @ y_new - b @ np.log(y_new) <= f - 0.25 * t * decrement or t < 1e-10:
                break
            t *= 0.5
        y = y_new
    return None


def risk_parity(cov: pd.DataFrame, budgets: np.ndarray | None = None) -> pd.Series:
    """Equal-risk-contribution (or risk-budget) portfolio via Spinu's convex programme."""
    S, _ = _as_arrays(cov)
    n = len(S)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    b = b / b.sum()
    # Rescale Sigma so the problem is well-conditioned regardless of units;
    # this rescales y but not w = y / sum(y).
    Sn = S / (np.trace(S) / n)
    y = _spinu_newton(Sn, b)
    if y is None:  # pragma: no cover - Newton essentially always converges here
        yv = cp.Variable(n, pos=True)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(yv, cp.psd_wrap(Sn)) - b @ cp.log(yv)))
        _solve(prob)
        y = yv.value
    return _finalize(y, cov.index)


def relaxed_risk_parity(mu: pd.Series, cov: pd.DataFrame, return_premium: float = 0.5,
                        lam: float = 0.0, budgets: np.ndarray | None = None,
                        min_weight: float = 0.0, max_weight: float = 1.0) -> pd.Series:
    r"""Relaxed risk parity (Gambeta & Kwon, 2020).

    .. math::
        \min_{x,\psi,\gamma,\rho}\ \psi - \gamma \quad \text{s.t.}\quad
        \zeta = \Sigma x,\quad x_i\zeta_i \ge n b_i\gamma^2,\quad
        \tfrac1n x^\top\Sigma x + \rho^2 \le \psi^2,\quad
        \lambda x^\top\Theta x \le \rho^2,\quad \mu^\top x \ge R,\quad
        \mathbf 1^\top x = 1,\ x \ge 0

    With equal budgets, :math:`\gamma \le \sigma/\sqrt n \le \psi`, so the
    objective is zero exactly at the ERC portfolio. The target ``R`` is set
    between the ERC return and the maximum attainable return:

    .. math:: R = r_{ERC} + \kappa\,(r_{max} - r_{ERC}), \quad \kappa = \text{return\_premium}

    ``lam`` (:math:`\lambda`) with :math:`\Theta = \mathrm{diag}(\Sigma)` adds the
    paper's optional regularisation; ``lam = 0`` is the pure formulation.
    """
    if not 0.0 <= return_premium < 1.0:
        # kappa = 1 forces the single max-return asset, where the cone constraints degenerate
        raise ValueError("return_premium must lie in [0, 1)")
    S, m = _as_arrays(cov, mu)
    n = len(S)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float) / np.sum(budgets)

    w_erc = risk_parity(cov, b).values
    r_erc = float(m @ w_erc)
    r_max = max_return(pd.Series(m, index=cov.index), min_weight, max_weight)
    gap = max(r_max - r_erc, 0.0)

    scale = np.trace(S) / n  # numerical conditioning only; the argmin in x is unchanged
    Sn = S / scale
    L = np.linalg.cholesky(Sn + 1e-12 * np.eye(n))

    x = cp.Variable(n)
    psi = cp.Variable()
    gamma = cp.Variable(nonneg=True)
    rho = cp.Variable(nonneg=True)
    target = cp.Parameter()
    zeta = Sn @ x

    cons = _box(x, min_weight, max_weight)
    cons.append(m @ x >= target)
    cons.append(cp.SOC(psi, cp.hstack([L.T @ x / np.sqrt(n), rho])))
    cons += [cp.quad_over_lin(gamma, x[i]) <= zeta[i] / (n * b[i]) for i in range(n)]
    if lam > 0:
        theta = np.sqrt(lam * np.diag(Sn))
        cons.append(cp.norm(cp.multiply(theta, x)) <= rho)
    prob = cp.Problem(cp.Minimize(psi - gamma), cons)

    def attempt(kappa: float) -> np.ndarray | None:
        target.value = r_erc + kappa * gap
        try:
            _solve(prob)
        except OptimizationError:
            return None
        return x.value.copy()

    sol = attempt(return_premium)
    if sol is None:
        # The cone constraints x_i (Sigma x)_i >= n b_i gamma^2 require every marginal
        # risk to be non-negative; with negatively correlated assets a high return
        # target can be unattainable under them. Use the largest feasible premium.
        lo_k, hi_k, sol = 0.0, return_premium, w_erc
        for _ in range(8):
            mid = 0.5 * (lo_k + hi_k)
            trial = attempt(mid)
            if trial is None:
                hi_k = mid
            else:
                lo_k, sol = mid, trial
        log.debug("RRP: return premium %.3f infeasible, using %.3f", return_premium, lo_k)
    return _finalize(sol, cov.index, min_weight)
