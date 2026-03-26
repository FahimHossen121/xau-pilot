from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from trading_bot.config import Settings
from trading_bot.data import get_candles_range
from trading_bot.htf_ai import build_live_controller
from trading_bot.htf_ai_collect import (
    build_historical_technical_snapshots,
    COLLECTION_COLUMNS,
    iter_historical_ai_replay_rows,
    load_collection_resume_state,
)


DEFAULT_WARMUP_DAYS = 10
DEFAULT_SECONDS_BETWEEN_AI_CALLS = 7.0


def _parse_date_label(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one reusable HTF AI replay CSV over a fixed window. "
            "By default, Gemini is only called on startup and technical HTF shifts."
        )
    )
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive ISO start date/time, for example 2026-01-01 or 2026-01-01T00:00:00",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Exclusive ISO end date/time, for example 2026-03-27 or 2026-03-27T00:00:00",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to reports/<symbol>_ai_replay_<window>_<mode>.csv",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=DEFAULT_WARMUP_DAYS,
        help="Warmup history to fetch before start-date so HTF indicators initialize cleanly",
    )
    parser.add_argument(
        "--include-expiry-refresh",
        action="store_true",
        help="Also refresh Gemini when the cached AI state expires during the historical sweep",
    )
    parser.add_argument(
        "--seconds-between-ai-calls",
        type=float,
        default=DEFAULT_SECONDS_BETWEEN_AI_CALLS,
        help="Pause between saved Gemini calls to stay under free-tier request-per-minute limits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial output CSV instead of starting over",
    )
    parser.add_argument(
        "--max-ai-calls",
        type=int,
        default=None,
        help="Optional hard cap on Gemini calls for this run, useful on free-tier quotas",
    )
    return parser


def _default_output_path(
    *,
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    include_expiry_refresh: bool,
) -> Path:
    mode_label = "shift_expiry" if include_expiry_refresh else "shift_only"
    return (
        Path("reports")
        / f"{symbol.lower()}_ai_replay_{start_time:%Y%m%d}_to_{end_time:%Y%m%d}_{mode_label}.csv"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    symbol = args.symbol or settings.symbol
    start_time = _parse_date_label(args.start_date)
    end_time = _parse_date_label(args.end_date)
    if start_time >= end_time:
        raise ValueError("start-date must be earlier than end-date.")

    warmup_start = start_time - timedelta(days=args.warmup_days)
    candles = get_candles_range(symbol, mt5.TIMEFRAME_M5, warmup_start, end_time)
    if len(candles) > 1:
        candles = candles.iloc[:-1]

    snapshots = build_historical_technical_snapshots(
        candles,
        start_time=start_time,
        end_time=end_time,
        rule="1H",
    )
    controller = build_live_controller(settings)

    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            include_expiry_refresh=args.include_expiry_refresh,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    resume_state = None
    resume_after = None
    existing_rows = 0
    write_mode = "w"
    write_header = True
    if output_path.exists():
        if args.resume:
            resume_state, resume_after, existing_rows = load_collection_resume_state(output_path)
            write_mode = "a"
            write_header = False
        elif not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Pass --overwrite to replace it."
            )

    rows_written_this_run = 0
    try:
        with output_path.open(write_mode, encoding="utf-8", newline="") as handle:
            for row, _ in iter_historical_ai_replay_rows(
                symbol=symbol,
                snapshots=snapshots,
                controller=controller,
                news_provider=settings.news_provider,
                gemini_model=settings.gemini_model,
                allow_expiry_refresh=args.include_expiry_refresh,
                seconds_between_ai_calls=max(0.0, args.seconds_between_ai_calls),
                initial_state=resume_state,
                skip_until=resume_after,
                max_ai_calls=args.max_ai_calls,
            ):
                pd.DataFrame([row], columns=COLLECTION_COLUMNS).to_csv(
                    handle,
                    index=False,
                    header=write_header,
                )
                handle.flush()
                write_header = False
                rows_written_this_run += 1
    except Exception:
        print("Historical AI Replay Collection")
        print(f"Symbol: {symbol}")
        print(f"Output CSV: {output_path}")
        print(f"Existing rows before this run: {existing_rows}")
        print(f"Rows written this run before failure: {rows_written_this_run}")
        raise

    print("Historical AI Replay Collection")
    print(f"Symbol: {symbol}")
    print(f"Window: {start_time.isoformat()} -> {end_time.isoformat()} (exclusive end)")
    print(f"Warmup start: {warmup_start.isoformat()}")
    print(f"H1 snapshots scanned: {len(snapshots)}")
    print(f"Existing rows before this run: {existing_rows}")
    print(f"Gemini checks saved this run: {rows_written_this_run}")
    print(f"Total saved rows: {existing_rows + rows_written_this_run}")
    print(f"Trigger mode: {'shift+expiry' if args.include_expiry_refresh else 'shift-only'}")
    print(f"Seconds between AI calls: {max(0.0, args.seconds_between_ai_calls):.1f}")
    print(f"News provider: {settings.news_provider}")
    print(f"Gemini model: {settings.gemini_model}")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()
