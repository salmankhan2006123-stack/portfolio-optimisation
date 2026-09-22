"""
main.py
-------
End-to-end pipeline: pulls stock data, computes MPT statistics, optimises
for maximum Sharpe ratio and minimum volatility, builds the efficient
frontier, and produces a summary plot.

Usage:
    python src/main.py
    python src/main.py --tickers AAPL MSFT GOOGL AMZN JPM --start 2021-01-01
"""

from __future__ import annotations

import argparse
import os

from data import compute_annualised_statistics, compute_daily_returns, download_price_data
from optimise import (
    RISK_FREE_RATE,
    efficient_frontier,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    random_portfolios,
)
from visualise import plot_efficient_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio optimisation using Modern Portfolio Theory")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"],
        help="Stock ticker symbols to include in the portfolio",
    )
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=RISK_FREE_RATE,
        help="Annualised risk-free rate used in the Sharpe ratio",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "output", "efficient_frontier.png"),
        help="Path to save the efficient frontier plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Downloading price data for {args.tickers} from {args.start}...")
    prices = download_price_data(args.tickers, start=args.start, end=args.end)
    daily_returns = compute_daily_returns(prices)
    expected_returns, volatilities, cov_matrix = compute_annualised_statistics(daily_returns)

    print("\nAnnualised expected returns:")
    print(expected_returns.round(4))
    print("\nAnnualised volatilities:")
    print(volatilities.round(4))

    print("\nOptimising for maximum Sharpe ratio...")
    max_sharpe = max_sharpe_portfolio(expected_returns, cov_matrix, args.risk_free_rate)
    print("Max Sharpe portfolio weights:")
    print(max_sharpe["weights"].round(4))
    print(
        f"Expected return: {max_sharpe['return']:.2%}, "
        f"Volatility: {max_sharpe['volatility']:.2%}, "
        f"Sharpe ratio: {max_sharpe['sharpe_ratio']:.3f}"
    )

    print("\nOptimising for minimum volatility...")
    min_vol = min_volatility_portfolio(expected_returns, cov_matrix)
    print("Min volatility portfolio weights:")
    print(min_vol["weights"].round(4))
    print(f"Expected return: {min_vol['return']:.2%}, Volatility: {min_vol['volatility']:.2%}")

    print("\nBuilding efficient frontier...")
    frontier = efficient_frontier(expected_returns, cov_matrix)

    print("Simulating random portfolios for context...")
    random_ports = random_portfolios(
        expected_returns, cov_matrix, risk_free_rate=args.risk_free_rate
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_efficient_frontier(
        frontier,
        random_ports,
        max_sharpe,
        min_vol,
        expected_returns,
        volatilities,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
