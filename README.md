# Portfolio Optimisation — Modern Portfolio Theory

A Python implementation of Modern Portfolio Theory (MPT) that automates stock
data collection, computes risk/return statistics, and finds optimal asset
allocations by maximising the Sharpe ratio.

## Features

- **Automated data collection** — pulls historical adjusted close prices for
  any set of tickers directly from Yahoo Finance (`yfinance`).
- **Risk/return statistics** — calculates annualised expected returns,
  volatilities, and the full covariance matrix of asset returns.
- **Sharpe ratio maximisation** — uses `scipy.optimize` (SLSQP) to solve for
  the long-only, fully-invested portfolio weights that maximise risk-adjusted
  return.
- **Minimum volatility portfolio** — solves for the lowest-risk portfolio on
  the frontier.
- **Efficient frontier** — traces the full frontier of optimal portfolios
  across the achievable return range.
- **Visualisation** — plots the efficient frontier against thousands of
  randomly simulated portfolios, individual assets, and the optimal
  portfolios, coloured by Sharpe ratio.

## Example output

Running the pipeline on `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM` produces a
chart like this (the classic MPT "bullet" curve):

- Grey/coloured dots — thousands of randomly weighted portfolios
- Black line — the efficient frontier
- Red diamonds — individual assets
- Gold star — the max-Sharpe-ratio portfolio
- Blue star — the minimum-volatility portfolio

## Project structure

```
portfolio-optimisation/
├── src/
│   ├── data.py         # Data collection & return/volatility/covariance stats
│   ├── optimise.py     # Sharpe maximisation, min-vol, and frontier logic
│   ├── visualise.py    # Efficient frontier plotting
│   └── main.py         # End-to-end pipeline entry point
├── output/              # Generated plots land here
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/portfolio-optimisation.git
cd portfolio-optimisation
pip install -r requirements.txt
```

## Usage

Run with the default tickers (AAPL, MSFT, GOOGL, AMZN, JPM):

```bash
python src/main.py
```

Or specify your own portfolio and date range:

```bash
python src/main.py --tickers AAPL TSLA NVDA JPM XOM --start 2019-01-01 --end 2025-01-01
```

Optional arguments:

| Flag | Description | Default |
|---|---|---|
| `--tickers` | List of stock tickers | `AAPL MSFT GOOGL AMZN JPM` |
| `--start` | Start date (YYYY-MM-DD) | `2021-01-01` |
| `--end` | End date (YYYY-MM-DD) | today |
| `--risk-free-rate` | Annualised risk-free rate for Sharpe ratio | `0.04` |
| `--output` | Path to save the plot | `output/efficient_frontier.png` |

The script prints the expected returns, volatilities, optimal weights for
both the max-Sharpe and minimum-volatility portfolios, and saves the
efficient frontier plot to `output/`.

## Methodology

1. **Data collection**: daily adjusted close prices are pulled via
   `yfinance` and converted to simple daily returns.
2. **Annualisation**: daily statistics are scaled to annual figures assuming
   252 trading days per year.
3. **Optimisation**: portfolio weights are constrained to be non-negative and
   sum to 1 (long-only, fully invested). `scipy.optimize.minimize` (SLSQP)
   solves:
   - **Max Sharpe**: minimise the negative Sharpe ratio
     `-(E[R_p] - R_f) / σ_p`
   - **Min volatility**: minimise `σ_p = sqrt(wᵀΣw)`
   - **Efficient frontier**: minimise `σ_p` for a series of fixed target
     returns spanning the achievable range.
4. **Visualisation**: 5,000 random long-only portfolios are simulated to give
   visual context to the frontier, plotted alongside the optimal portfolios.

## Requirements

- Python 3.10+
- `yfinance`, `numpy`, `pandas`, `scipy`, `matplotlib`

## Notes

- This is an educational/portfolio project applying MPT concepts (Markowitz,
  1952) and is not investment advice.
- Data quality and availability depend on Yahoo Finance; some tickers or
  date ranges may return incomplete data.
