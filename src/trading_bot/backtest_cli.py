from __future__ import annotations

import argparse

import MetaTrader5 as mt5

from trading_bot.backtest import format_backtest_summary, run_mt5_ltf_backtest
from trading_bot.config import Settings

EXECUTION_TIMEFRAME = "M5"
HTF_RULE = "1H"
DEFAULT_CANDLE_COUNT = 3000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper-only MT5 backtest for the active strategy: M5 execution with H1 HTF."
    )
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CANDLE_COUNT,
        help="Number of M5 candles to fetch from MT5",
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
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=None,
        help="Daily realized loss fraction that blocks new trades for the rest of the day",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=3,
        help="Number of bars to wait after triggering a loss-streak cooldown",
    )
    parser.add_argument(
        "--cooldown-loss-streak",
        type=int,
        default=2,
        help="Consecutive losing trades required before a cooldown starts",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    result = run_mt5_ltf_backtest(
        symbol=args.symbol or settings.symbol,
        timeframe=mt5.TIMEFRAME_M5,
        count=args.count,
        initial_balance=args.balance,
        risk_fraction=args.risk if args.risk is not None else settings.max_risk_per_trade,
        spread=args.spread,
        slippage=args.slippage,
        max_daily_loss_fraction=(
            args.max_daily_loss if args.max_daily_loss is not None else settings.max_daily_loss
        ),
        cooldown_bars_after_loss=args.cooldown_bars,
        loss_streak_for_cooldown=args.cooldown_loss_streak,
        use_htf_filter=True,
        htf_rule=HTF_RULE,
    )
    print(format_backtest_summary(result))


if __name__ == "__main__":
    main()
