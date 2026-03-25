from __future__ import annotations

import argparse

import MetaTrader5 as mt5

from trading_bot.backtest import format_backtest_summary, run_mt5_ltf_backtest
from trading_bot.config import Settings

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a paper-only MT5 backtest.")
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument(
        "--timeframe",
        default="M15",
        choices=sorted(TIMEFRAME_MAP.keys()),
        help="MT5 candle timeframe",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Number of candles to fetch from MT5",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1000.0,
        help="Starting account balance for the backtest",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=None,
        help="Risk fraction per trade, default comes from .env",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.0,
        help="Absolute price spread assumption per round trip instrument quote",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Absolute price slippage assumption per side",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    result = run_mt5_ltf_backtest(
        symbol=args.symbol or settings.symbol,
        timeframe=TIMEFRAME_MAP[args.timeframe],
        count=args.count,
        initial_balance=args.balance,
        risk_fraction=args.risk if args.risk is not None else settings.max_risk_per_trade,
        spread=args.spread,
        slippage=args.slippage,
    )
    print(format_backtest_summary(result))


if __name__ == "__main__":
    main()
