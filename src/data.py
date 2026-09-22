"""
data.py
-------
Automates collection of historical stock price data via Yahoo Finance
and derives the statistical inputs needed for Modern Portfolio Theory:
annualised expected returns, volatilities, and the covariance matrix
of returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def download_price_data(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        Stock ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"].
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str, optional
        End date in "YYYY-MM-DD" format. Defaults to today if omitted.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices indexed by date, one column per ticker.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex column structure ("Close", ticker) when
    # multiple tickers are requested, and a flat structure for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()

    if prices.empty:
        raise ValueError(
            "No price data was returned. Check the ticker symbols and date range."
        )

    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


def compute_annualised_statistics(
    daily_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Annualise expected returns, volatilities, and the covariance matrix
    from daily return data.

    Returns
    -------
    expected_returns : pd.Series
        Annualised mean return per asset.
    volatilities : pd.Series
        Annualised standard deviation (volatility) per asset.
    cov_matrix : pd.DataFrame
        Annualised covariance matrix of asset returns.
    """
    expected_returns = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = daily_returns.cov() * TRADING_DAYS_PER_YEAR
    volatilities = np.sqrt(np.diag(cov_matrix))
    volatilities = pd.Series(volatilities, index=daily_returns.columns)

    return expected_returns, volatilities, cov_matrix


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    prices = download_price_data(tickers, start="2021-01-01")
    daily_returns = compute_daily_returns(prices)
    expected_returns, volatilities, cov_matrix = compute_annualised_statistics(
        daily_returns
    )

    print("Annualised expected returns:\n", expected_returns, "\n")
    print("Annualised volatilities:\n", volatilities, "\n")
    print("Annualised covariance matrix:\n", cov_matrix)
