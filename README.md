# Trading Bot Template (Python 3.11, MT5, Paper Mode)

This repository is an industry-style starter template for building a trading bot.
It is intentionally scaffold-only: no real strategy implementation is included yet.

## Stack

- Python 3.11
- venv + pip
- MetaTrader 5 Python integration
- Paper-trading first workflow
- GitHub-compatible CI and quality checks

## Quick Start

1. Install Python 3.11.
2. Create and activate virtual environment:

   Windows PowerShell:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

4. Set environment variables:

   ```powershell
   copy .env.example .env
   ```

5. Fill in `.env` values for your MT5 demo account and Gemini key.

6. Run smoke tests:

   ```powershell
   pytest
   ```

## Development Rules

- Start in paper mode only.
- Keep `ENABLE_TRADING=false` until explicit go-live criteria are met.
- Treat broker credentials and API keys as secrets.

## Project Layout

- `src/trading_bot/` core package
- `tests/` test suite
- `docs/` architecture and runbook docs
- `.github/workflows/` CI checks

## Next Step

You will implement logic incrementally in this order:

1. Config loading and validation
2. MT5 connectivity wrapper
3. Data ingestion and normalization
4. Signal pipeline
5. Risk engine
6. Paper execution simulator
7. Reporting and alerts
