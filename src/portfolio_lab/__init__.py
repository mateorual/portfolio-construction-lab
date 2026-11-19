"""portfolio_lab: an out-of-sample laboratory for portfolio construction methods.

Modules
-------
config         YAML experiment configuration -> typed dataclasses
data           price / risk-free download with a local parquet cache
estimators     covariance and expected-return estimators
black_litterman  equilibrium returns, views and the Bayesian posterior
optimizers     convex portfolio optimizers (min-variance, mean-variance, ERC, RRP)
strategies     strategy definitions that turn estimates into target weights
backtest       event-driven monthly rebalancing engine with drift and costs
metrics        performance, diversification and bootstrap statistics
plotting       figure generation with a consistent visual style
experiments    the research pipeline (backtest, ablations, robustness)
"""

__version__ = "0.1.0"
