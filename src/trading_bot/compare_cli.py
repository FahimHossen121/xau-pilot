from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_bot.backtest import (
    backtest_result_to_row,
    backtest_session_stats_to_frame,
    backtest_trades_to_frame,
    format_backtest_summary,
    run_mt5_ltf_backtest,
)
from trading_bot.config import Settings

SCENARIO_NAME = "xauusd_m5_execution_h1_htf"
EXECUTION_TIMEFRAME = "M5"
HTF_RULE = "1H"
DEFAULT_CANDLE_COUNT = 3000


def _parse_date_label(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def _build_report_stem(
    *,
    symbol: str,
    timeframe: str,
    htf_rule: str,
    timestamp: str,
    start_time: datetime | None,
    end_time: datetime | None,
) -> str:
    if start_time is not None and end_time is not None:
        window = f"{start_time:%Y%m%d}_to_{end_time:%Y%m%d}"
    else:
        window = "latest_window"
    return f"{symbol.lower()}_{timeframe.lower()}_{htf_rule.lower()}_{window}_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single active MT5 backtest scenario and export CSV reports."
    )
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance for each scenario")
    parser.add_argument("--risk", type=float, default=None, help="Risk fraction override")
    parser.add_argument("--spread", type=float, default=0.30, help="Absolute spread assumption")
    parser.add_argument("--slippage", type=float, default=0.05, help="Absolute slippage assumption per side")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CANDLE_COUNT,
        help="Number of M5 candles to fetch from MT5",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date/time in ISO format, for example 2026-01-01 or 2026-01-01T00:00:00",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Exclusive end date/time in ISO format, for example 2026-03-26 or 2026-03-25T23:59:59",
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
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where summary and trades CSV files will be written",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    symbol = args.symbol or settings.symbol
    risk_fraction = args.risk if args.risk is not None else settings.max_risk_per_trade
    max_daily_loss_fraction = (
        args.max_daily_loss if args.max_daily_loss is not None else settings.max_daily_loss
    )
    start_time = _parse_date_label(args.start_date)
    end_time = _parse_date_label(args.end_date)
    if (start_time is None) != (end_time is None):
        raise ValueError("start-date and end-date must be provided together.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: list[dict[str, float | int | str | bool | None]] = []
    trade_frames: list[pd.DataFrame] = []
    session_frames: list[pd.DataFrame] = []

    import MetaTrader5 as mt5

    result = run_mt5_ltf_backtest(
        symbol=symbol,
        timeframe=mt5.TIMEFRAME_M5,
        count=args.count,
        initial_balance=args.balance,
        risk_fraction=risk_fraction,
        spread=args.spread,
        slippage=args.slippage,
        max_daily_loss_fraction=max_daily_loss_fraction,
        cooldown_bars_after_loss=args.cooldown_bars,
        loss_streak_for_cooldown=args.cooldown_loss_streak,
        start_time=start_time,
        end_time=end_time,
        use_htf_filter=True,
        htf_rule=HTF_RULE,
    )
    summary_rows.append(
        backtest_result_to_row(
            result,
            scenario_name=SCENARIO_NAME,
            symbol=symbol,
            timeframe=EXECUTION_TIMEFRAME,
            candle_count=args.count if start_time is None else -1,
            window_start=start_time.isoformat() if start_time is not None else None,
            window_end=end_time.isoformat() if end_time is not None else None,
            risk_fraction=risk_fraction,
            spread=args.spread,
            slippage=args.slippage,
        )
    )
    trade_frames.append(
        backtest_trades_to_frame(
            result,
            scenario_name=SCENARIO_NAME,
            symbol=symbol,
            timeframe=EXECUTION_TIMEFRAME,
        )
    )
    session_frames.append(
        backtest_session_stats_to_frame(
            result,
            scenario_name=SCENARIO_NAME,
            symbol=symbol,
            timeframe=EXECUTION_TIMEFRAME,
        )
    )
    print()
    print(f"[{SCENARIO_NAME}]")
    print(format_backtest_summary(result))

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    sessions_df = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()

    stem = _build_report_stem(
        symbol=symbol,
        timeframe=EXECUTION_TIMEFRAME,
        htf_rule=HTF_RULE,
        timestamp=timestamp,
        start_time=start_time,
        end_time=end_time,
    )
    summary_path = output_dir / f"{stem}_summary.csv"
    trades_path = output_dir / f"{stem}_trades.csv"
    sessions_path = output_dir / f"{stem}_sessions.csv"
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    sessions_df.to_csv(sessions_path, index=False)

    print()
    print(f"Summary CSV: {summary_path}")
    print(f"Trades CSV: {trades_path}")
    print(f"Sessions CSV: {sessions_path}")


if __name__ == "__main__":
    main()
