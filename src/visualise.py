"""
visualise.py
------------
Plots the efficient frontier alongside randomly simulated portfolios,
individual assets, and the highlighted optimal (max-Sharpe) and
minimum-volatility portfolios.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    random_ports: pd.DataFrame,
    max_sharpe: dict,
    min_vol: dict,
    expected_returns: pd.Series,
    volatilities: pd.Series,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Simulated random portfolios, coloured by Sharpe ratio
    scatter = ax.scatter(
        random_ports["volatility"],
        random_ports["return"],
        c=random_ports["sharpe"],
        cmap="viridis",
        alpha=0.4,
        s=8,
        label="Simulated portfolios",
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")

    # Efficient frontier line
    ax.plot(
        frontier["volatility"],
        frontier["return"],
        color="black",
        linewidth=2,
        label="Efficient frontier",
    )

    # Individual assets
    ax.scatter(
        volatilities,
        expected_returns,
        marker="D",
        color="red",
        s=60,
        label="Individual assets",
    )
    for ticker in expected_returns.index:
        ax.annotate(
            ticker,
            (volatilities[ticker], expected_returns[ticker]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    # Max Sharpe portfolio
    ax.scatter(
        max_sharpe["volatility"],
        max_sharpe["return"],
        marker="*",
        color="gold",
        edgecolors="black",
        s=400,
        label="Max Sharpe ratio",
        zorder=5,
    )

    # Min volatility portfolio
    ax.scatter(
        min_vol["volatility"],
        min_vol["return"],
        marker="*",
        color="deepskyblue",
        edgecolors="black",
        s=400,
        label="Min volatility",
        zorder=5,
    )

    ax.set_title("Efficient Frontier — Modern Portfolio Theory")
    ax.set_xlabel("Annualised Volatility (Risk)")
    ax.set_ylabel("Annualised Expected Return")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()
