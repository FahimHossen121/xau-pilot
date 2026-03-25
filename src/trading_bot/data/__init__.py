"""Market data helpers."""

from __future__ import annotations

from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd


def _normalize_rates(rates, symbol: str) -> pd.DataFrame:
    if rates is None:
        raise RuntimeError(f"No data returned for {symbol}")

    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"No candle rows returned for {symbol}")

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume"]]


def get_candles(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    """Fetch recent candles from MT5 and return a normalized OHLCV frame."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    finally:
        mt5.shutdown()

    return _normalize_rates(rates, symbol)


def get_candles_range(
    symbol: str,
    timeframe: int,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Fetch a fixed historical candle window from MT5."""
    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time.")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
    finally:
        mt5.shutdown()

    return _normalize_rates(rates, symbol)
