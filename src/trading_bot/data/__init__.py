"""Market data helpers."""

from __future__ import annotations

import MetaTrader5 as mt5
import pandas as pd


def get_candles(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
	"""Fetch recent candles from MT5 and return a normalized OHLCV frame."""
	if not mt5.initialize():
		raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

	try:
		rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
	finally:
		mt5.shutdown()

	if rates is None:
		raise RuntimeError(f"No data returned for {symbol}")

	df = pd.DataFrame(rates)
	df["time"] = pd.to_datetime(df["time"], unit="s")
	df.set_index("time", inplace=True)
	return df[["open", "high", "low", "close", "tick_volume"]]
