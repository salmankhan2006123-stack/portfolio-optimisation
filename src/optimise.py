"""
optimise.py
-----------
Implements Modern Portfolio Theory optimisation:
- Sharpe ratio maximisation via scipy.optimize (SLSQP)
- Minimum-volatility portfolio
- Efficient frontier construction
- Visualisation of the frontier, individual assets, and the optimal
  (max-Sharpe) and minimum-volatility portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RISK_FREE_RATE = 0.04  # annualised, adjust to current risk-free rate


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple[float, float]:
    """Return (expected annual return, annual volatility) for a weight vector."""
    port_return = float(np.dot(weights, expected_returns))
    port_vol = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    return port_return, port_vol


def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Objective function: negative Sharpe ratio (scipy.optimize minimises)."""
    port_return, port_vol = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(port_return - risk_free_rate) / port_vol


def portfolio_volatility(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> float:
    """Objective function used for minimum-volatility and frontier optimisation."""
    return portfolio_performance(weights, expected_returns, cov_matrix)[1]


def _base_constraints(n_assets: int):
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # long-only, no leverage
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    initial_guess = np.repeat(1.0 / n_assets, n_assets)
    return bounds, constraints, initial_guess


def max_sharpe_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Find portfolio weights that maximise the Sharpe ratio."""
    n_assets = len(expected_returns)
    bounds, constraints, initial_guess = _base_constraints(n_assets)

    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Sharpe ratio optimisation failed: {result.message}")

    weights = result.x
    port_return, port_vol = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe = (port_return - risk_free_rate) / port_vol

    return {
        "weights": pd.Series(weights, index=expected_returns.index),
        "return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe,
    }


def min_volatility_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict:
    """Find portfolio weights that minimise volatility."""
    n_assets = len(expected_returns)
    bounds, constraints, initial_guess = _base_constraints(n_assets)

    result = minimize(
        portfolio_volatility,
        initial_guess,
        args=(expected_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Minimum volatility optimisation failed: {result.message}")

    weights = result.x
    port_return, port_vol = portfolio_performance(weights, expected_returns, cov_matrix)

    return {
        "weights": pd.Series(weights, index=expected_returns.index),
        "return": port_return,
        "volatility": port_vol,
    }


def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by minimising volatility for a range of
    target returns spanning the achievable range.

    Returns
    -------
    pd.DataFrame with columns ["return", "volatility"], one row per target.
    """
    n_assets = len(expected_returns)
    bounds, _, initial_guess = _base_constraints(n_assets)

    target_returns = np.linspace(
        expected_returns.min(), expected_returns.max(), n_points
    )
    frontier = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, target=target: portfolio_performance(
                    w, expected_returns, cov_matrix
                )[0]
                - target,
            },
        )
        result = minimize(
            portfolio_volatility,
            initial_guess,
            args=(expected_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            frontier.append({"return": target, "volatility": result.fun})

    return pd.DataFrame(frontier)


def random_portfolios(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate random long-only portfolios for scatter-plot context behind
    the efficient frontier.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(expected_returns)
    records = []

    for _ in range(n_portfolios):
        weights = rng.random(n_assets)
        weights /= weights.sum()
        port_return, port_vol = portfolio_performance(weights, expected_returns, cov_matrix)
        sharpe = (port_return - risk_free_rate) / port_vol
        records.append({"return": port_return, "volatility": port_vol, "sharpe": sharpe})

    return pd.DataFrame(records)
