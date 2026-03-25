from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5

from trading_bot.config import Settings
from trading_bot.data import get_candles, get_candles_range
from trading_bot.htf_ai import (
    build_live_controller,
    build_technical_htf_snapshot,
    load_htf_ai_state,
    save_htf_ai_state,
)


DEFAULT_CANDLE_COUNT = 2000
DEFAULT_CACHE_PATH = Path("state") / "htf_ai_state.json"


def _parse_date_label(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the H1 technical state and confirm it through Brave news + Gemini."
    )
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CANDLE_COUNT,
        help="Number of M5 candles to fetch when no date range is provided",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive ISO start date/time for a fixed evaluation window",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Exclusive ISO end date/time for a fixed evaluation window",
    )
    parser.add_argument(
        "--cache-path",
        default=str(DEFAULT_CACHE_PATH),
        help="JSON file used to persist the latest AI-confirmed HTF state",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force an AI refresh even if the cached state has not expired",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    symbol = args.symbol or settings.symbol

    start_time = _parse_date_label(args.start_date)
    end_time = _parse_date_label(args.end_date)
    if (start_time is None) != (end_time is None):
        raise ValueError("start-date and end-date must be provided together.")

    candles = (
        get_candles_range(symbol, mt5.TIMEFRAME_M5, start_time, end_time)
        if start_time is not None and end_time is not None
        else get_candles(symbol, mt5.TIMEFRAME_M5, args.count)
    )
    if len(candles) > 1:
        candles = candles.iloc[:-1]

    technical_snapshot = build_technical_htf_snapshot(candles, rule="1H")
    state_path = Path(args.cache_path)
    current_state = load_htf_ai_state(state_path)
    controller = build_live_controller(settings)
    evaluation = controller.evaluate(
        symbol=symbol,
        technical_snapshot=technical_snapshot,
        state=current_state,
        now=technical_snapshot.as_of,
        force_refresh=args.force_refresh,
    )
    save_htf_ai_state(state_path, evaluation.state)

    print("HTF AI Evaluation")
    print(f"Symbol: {symbol}")
    print(f"As of (UTC): {technical_snapshot.as_of.isoformat()}")
    print(f"Technical H1 state: {technical_snapshot.current_state.value}")
    print(f"Previous technical H1 state: {technical_snapshot.previous_state.value if technical_snapshot.previous_state else 'none'}")
    print(f"Technical changed: {technical_snapshot.changed}")
    print(f"Trigger reason: {evaluation.trigger_reason}")
    print(f"AI called: {evaluation.ai_called}")
    print(f"Articles used: {evaluation.article_count}")
    print(f"Trading enabled: {evaluation.state.trading_enabled}")
    print(f"Effective state: {evaluation.state.effective_state.value if evaluation.state.effective_state else 'none'}")
    print(f"AI state: {evaluation.state.ai_state.value if evaluation.state.ai_state else 'none'}")
    print(f"AI confidence: {evaluation.state.ai_confidence if evaluation.state.ai_confidence is not None else 'n/a'}")
    print(f"AI checked at: {evaluation.state.ai_checked_at.isoformat() if evaluation.state.ai_checked_at else 'n/a'}")
    print(f"AI expires at: {evaluation.state.ai_expires_at.isoformat() if evaluation.state.ai_expires_at else 'n/a'}")
    print(f"Stop reason: {evaluation.state.stop_reason or 'none'}")
    if evaluation.state.ai_summary:
        print(f"AI summary: {evaluation.state.ai_summary}")
    if evaluation.state.drivers:
        print("Drivers:")
        for item in evaluation.state.drivers:
            print(f"- {item}")
    if evaluation.state.invalidates:
        print("Invalidates:")
        for item in evaluation.state.invalidates:
            print(f"- {item}")
    print(f"State file: {state_path}")


if __name__ == "__main__":
    main()
