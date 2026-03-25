"""Technical indicator helpers for strategy development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = {"open", "high", "low", "close"}
REQUIRED_INDICATOR_COLUMNS = {"ema_50", "ema_200", "rsi_14", "atr_14"}
DEFAULT_LTF_WEIGHTS = {
    "ema": 0.35,
    "rsi": 0.25,
    "structure": 0.40,
}
DEFAULT_LTF_THRESHOLD = 0.40
DEFAULT_ATR_FLOOR_RATIO = 0.0010
DEFAULT_STRUCTURE_LOOKBACK = 20


@dataclass(frozen=True)
class LTFSignal:
    score: float
    bias: str
    tradable: bool
    threshold: float
    atr_ratio: float
    component_scores: dict[str, float]
    reason: str | None = None


def _validate_price_frame(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_PRICE_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _validate_indicator_frame(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_INDICATOR_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required indicator columns: {missing}")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the price frame with core trend and volatility indicators."""
    _validate_price_frame(df)

    frame = df.copy()

    frame["ema_50"] = frame["close"].ewm(span=50, adjust=False).mean()
    frame["ema_200"] = frame["close"].ewm(span=200, adjust=False).mean()

    delta = frame["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    frame["rsi_14"] = 100 - (100 / (1 + rs))
    frame["rsi_14"] = frame["rsi_14"].fillna(100.0)

    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift()).abs()
    low_close = (frame["low"] - frame["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    frame["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    return frame


def add_ltf_features(
    df: pd.DataFrame,
    *,
    structure_lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a copy of the frame with normalized component signals and LTF score."""
    if structure_lookback < 2:
        raise ValueError("structure_lookback must be at least 2.")

    frame = add_indicators(df)
    _validate_indicator_frame(frame)

    active_weights = weights or DEFAULT_LTF_WEIGHTS
    missing_weights = {"ema", "rsi", "structure"}.difference(active_weights)
    if missing_weights:
        missing = ", ".join(sorted(missing_weights))
        raise ValueError(f"weights are missing required keys: {missing}")

    frame["ema_signal"] = np.select(
        [
            (frame["close"] >= frame["ema_50"]) & (frame["ema_50"] >= frame["ema_200"]),
            (frame["close"] <= frame["ema_50"]) & (frame["ema_50"] <= frame["ema_200"]),
        ],
        [1.0, -1.0],
        default=0.0,
    )

    frame["rsi_signal"] = ((frame["rsi_14"] - 50.0) / 50.0).clip(-1.0, 1.0)

    prior_high = frame["high"].shift(1).rolling(window=structure_lookback).max()
    prior_low = frame["low"].shift(1).rolling(window=structure_lookback).min()
    frame["structure_signal"] = np.select(
        [
            frame["close"] > prior_high,
            frame["close"] < prior_low,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    frame["atr_ratio"] = (frame["atr_14"] / frame["close"]).fillna(0.0)
    frame["ltf_score"] = (
        active_weights["ema"] * frame["ema_signal"]
        + active_weights["rsi"] * frame["rsi_signal"]
        + active_weights["structure"] * frame["structure_signal"]
    )

    return frame


def get_latest_ltf_signal(
    df: pd.DataFrame,
    *,
    threshold: float = DEFAULT_LTF_THRESHOLD,
    atr_floor_ratio: float = DEFAULT_ATR_FLOOR_RATIO,
    structure_lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
    weights: dict[str, float] | None = None,
) -> LTFSignal:
    """Score the latest candle and return a deterministic LTF decision payload."""
    feature_frame = add_ltf_features(
        df,
        structure_lookback=structure_lookback,
        weights=weights,
    )
    latest = feature_frame.iloc[-1]

    component_scores = {
        "ema": float(latest["ema_signal"]),
        "rsi": float(latest["rsi_signal"]),
        "structure": float(latest["structure_signal"]),
    }
    atr_ratio = float(latest["atr_ratio"])
    score = float(latest["ltf_score"])

    if atr_ratio < atr_floor_ratio:
        return LTFSignal(
            score=score,
            bias="neutral",
            tradable=False,
            threshold=threshold,
            atr_ratio=atr_ratio,
            component_scores=component_scores,
            reason="atr_below_floor",
        )

    if score >= threshold:
        bias = "bullish"
    elif score <= -threshold:
        bias = "bearish"
    else:
        bias = "neutral"

    return LTFSignal(
        score=score,
        bias=bias,
        tradable=True,
        threshold=threshold,
        atr_ratio=atr_ratio,
        component_scores=component_scores,
        reason=None,
    )
