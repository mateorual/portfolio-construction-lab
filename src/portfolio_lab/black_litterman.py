r"""Black-Litterman (1992) model, implemented from first principles.

Notation (all annualised):

* :math:`\Sigma`  - asset covariance
* :math:`w_m`     - reference ("market") weights
* :math:`\delta`  - risk aversion of the representative investor
* :math:`\pi = \delta \Sigma w_m` - implied equilibrium excess returns (reverse optimisation)
* :math:`P, Q`    - K views: :math:`P\mu = Q + \varepsilon,\ \varepsilon \sim N(0, \Omega)`
* :math:`\tau`    - scales the uncertainty of the prior mean, :math:`\mu \sim N(\pi, \tau\Sigma)`

Posterior (Theil mixed estimation):

.. math::
    \mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}
               \left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\right]

    \Sigma_{BL} = \Sigma + \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}

The posterior is then fed to a *return-aware* optimizer (mean-variance). Feeding
it to a minimum-variance optimizer - a common mistake - silently discards the
views, because minimum variance never looks at expected returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BlackLittermanResult:
    prior: pd.Series  # pi
    posterior: pd.Series  # mu_BL
    posterior_cov: pd.DataFrame  # Sigma_BL
    views: pd.Series  # Q (indexed by asset for absolute views)


def implied_equilibrium_returns(cov: pd.DataFrame, market_weights: pd.Series,
                                risk_aversion: float) -> pd.Series:
    """Reverse optimisation: the excess returns that make ``market_weights`` mean-variance optimal."""
    w = market_weights.reindex(cov.index).values
    return pd.Series(risk_aversion * cov.values @ w, index=cov.index, name="pi")


def he_litterman_omega(P: np.ndarray, cov: np.ndarray, tau: float,
                       confidence: float = 1.0) -> np.ndarray:
    """View uncertainty proportional to the prior variance of each view portfolio.

    ``confidence`` > 1 tightens the views, < 1 loosens them (Omega / confidence).
    """
    return np.diag(np.diag(P @ (tau * cov) @ P.T)) / confidence


def momentum_views(scores: pd.Series, cov: pd.DataFrame, prior: pd.Series,
                   information_coefficient: float, clip_z: float = 3.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Absolute views from a cross-sectional momentum signal (Grinold's alpha rule).

    ``alpha_i = IC * sigma_i * z_i`` - the expected excess return of a signal with
    information coefficient IC, scaled by the asset's own volatility. The view on
    each asset is the prior plus that alpha, so a zero signal leaves the prior intact.
    """
    s = scores.reindex(cov.index)
    dispersion = s.std(ddof=0)
    if not np.isfinite(dispersion) or dispersion < 1e-12:
        # no cross-sectional signal (dividing float noise by ~1e-17 would invent views)
        z = pd.Series(0.0, index=s.index)
    else:
        z = ((s - s.mean()) / dispersion).clip(-clip_z, clip_z).fillna(0.0)
    vol = np.sqrt(np.diag(cov.values))
    alpha = information_coefficient * vol * z.values
    P = np.eye(len(cov))
    Q = prior.values + alpha
    return P, Q


def posterior(cov: pd.DataFrame, prior: pd.Series, P: np.ndarray, Q: np.ndarray,
              tau: float, omega: np.ndarray) -> BlackLittermanResult:
    S = cov.values
    tS_inv = np.linalg.inv(tau * S)
    if P.size == 0:  # no views: posterior mean is the prior
        post_prec = tS_inv
        rhs = tS_inv @ prior.values
    else:
        omega_inv = np.linalg.inv(omega)
        post_prec = tS_inv + P.T @ omega_inv @ P
        rhs = tS_inv @ prior.values + P.T @ omega_inv @ Q
    M = np.linalg.inv(post_prec)
    mu = M @ rhs
    cov_post = S + M
    cov_post = (cov_post + cov_post.T) / 2.0
    idx = cov.index
    absolute_views = P.size and P.shape == (len(idx), len(idx)) and np.allclose(P, np.eye(len(idx)))
    view_series = pd.Series(Q, index=idx if absolute_views else None, name="Q") if P.size \
        else pd.Series(dtype=float, name="Q")
    return BlackLittermanResult(
        prior=prior.rename("pi"),
        posterior=pd.Series(mu, index=idx, name="mu_bl"),
        posterior_cov=pd.DataFrame(cov_post, index=idx, columns=idx),
        views=view_series,
    )


def black_litterman_momentum(cov: pd.DataFrame, market_weights: pd.Series,
                             momentum: pd.Series, risk_aversion: float, tau: float,
                             information_coefficient: float,
                             confidence: float = 1.0) -> BlackLittermanResult:
    """Convenience wrapper: market prior + momentum views -> posterior."""
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)
    P, Q = momentum_views(momentum, cov, pi, information_coefficient)
    omega = he_litterman_omega(P, cov.values, tau, confidence)
    return posterior(cov, pi, P, Q, tau, omega)
