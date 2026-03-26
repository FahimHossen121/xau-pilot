from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5

from trading_bot.config import Settings
from trading_bot.data import get_candles_range
from trading_bot.htf_ai_replay import build_technical_seed_ai_history


def _parse_date_label(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a starter AI replay CSV from the H1 technical HTF classifier."
    )
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive ISO start date/time, for example 2026-03-16 or 2026-03-16T00:00:00",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive ISO end date/time, for example 2026-03-27 or 2026-03-26T23:59:59",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to reports/<symbol>_ai_seed_<window>.csv",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    symbol = args.symbol or settings.symbol
    start_time = _parse_date_label(args.start_date)
    end_time = _parse_date_label(args.end_date)
    if start_time >= end_time:
        raise ValueError("start-date must be earlier than end-date.")

    warmup_start = start_time - timedelta(days=10)
    candles = get_candles_range(symbol, mt5.TIMEFRAME_M5, warmup_start, end_time)
    if len(candles) > 1:
        candles = candles.iloc[:-1]

    seed_df = build_technical_seed_ai_history(
        candles,
        start_time=start_time,
        end_time=end_time,
        rule="1H",
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path("reports")
        / f"{symbol.lower()}_ai_seed_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(output_path, index=False)

    print(f"Rows written: {len(seed_df)}")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()
