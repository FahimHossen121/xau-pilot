"""Technical indicator helpers for strategy development."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


class HTFState(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


@dataclass(frozen=True)
class HTFPolicy:
    state: HTFState
    allow_long: bool
    allow_short: bool
    frequency_divisor: int
    note: str


@dataclass(frozen=True)
class HTFSignal:
    state: HTFState
    atr_ratio: float
    trend_score: float
    policy: HTFPolicy


class TradingSession(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


@dataclass(frozen=True)
class SessionProfile:
    session: TradingSession
    trend_threshold_multiplier: float
    atr_floor_multiplier: float
    range_edge: float
    range_buy_rsi: float
    range_sell_rsi: float
    range_threshold: float
    reward_to_risk: float
    atr_multiplier: float
    note: str


@dataclass(frozen=True)
class TradeDecision:
    bias: str
    tradable: bool
    strategy_mode: str
    session: TradingSession
    htf_state: str | None
    score: float
    threshold: float
    atr_ratio: float
    reward_to_risk: float
    atr_multiplier: float
    reason: str | None = None


def _validate_price_frame(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_PRICE_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")


def _normalize_resample_rule(rule: str) -> str:
    return rule.replace("H", "h")


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


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample lower-timeframe candles into a higher-timeframe OHLC frame."""
    _validate_price_frame(df)
    _validate_datetime_index(df)
    normalized_rule = _normalize_resample_rule(rule)

    resampled = df.resample(normalized_rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    return resampled.dropna()


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
    frame["range_high"] = prior_high
    frame["range_low"] = prior_low
    range_width = (frame["range_high"] - frame["range_low"]).replace(0, pd.NA)
    frame["range_position"] = (
        ((frame["close"] - frame["range_low"]) / range_width).clip(0.0, 1.0).fillna(0.5)
    )
    frame["ema_spread_ratio"] = (
        (frame["ema_50"] - frame["ema_200"]).abs() / frame["close"]
    ).fillna(0.0)

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


def get_trade_decision(
    df: pd.DataFrame,
    *,
    timestamp: pd.Timestamp | None = None,
    htf_state: HTFState | str | None = None,
    threshold: float = DEFAULT_LTF_THRESHOLD,
    atr_floor_ratio: float = DEFAULT_ATR_FLOOR_RATIO,
    structure_lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
    weights: dict[str, float] | None = None,
) -> TradeDecision:
    """Return a session-aware trade decision from the latest available LTF features."""
    feature_frame = add_ltf_features(
        df,
        structure_lookback=structure_lookback,
        weights=weights,
    )
    latest = feature_frame.iloc[-1]
    current_timestamp = timestamp or feature_frame.index[-1]
    session = get_trading_session(current_timestamp)
    profile = get_session_profile(session)

    current_htf_state = str(htf_state) if htf_state is not None else None
    atr_ratio = float(latest["atr_ratio"])
    effective_atr_floor = atr_floor_ratio * profile.atr_floor_multiplier

    if session is TradingSession.OFF_HOURS:
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="blocked",
            session=session,
            htf_state=current_htf_state,
            score=float(latest["ltf_score"]),
            threshold=threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="off_hours_blocked",
        )

    if current_htf_state == HTFState.VOLATILE.value:
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="blocked",
            session=session,
            htf_state=current_htf_state,
            score=0.0,
            threshold=threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="htf_volatile",
        )

    if current_htf_state != HTFState.SIDEWAYS.value and atr_ratio < effective_atr_floor:
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="blocked",
            session=session,
            htf_state=current_htf_state,
            score=float(latest["ltf_score"]),
            threshold=threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="atr_below_session_floor",
        )

    if current_htf_state == HTFState.SIDEWAYS.value:
        range_position = float(latest["range_position"])
        rsi_value = float(latest["rsi_14"])
        ema_neutral = float(latest["ema_spread_ratio"]) <= 0.0025

        long_edge = max(0.0, (profile.range_edge - range_position) / profile.range_edge)
        short_edge = max(
            0.0,
            (range_position - (1.0 - profile.range_edge)) / profile.range_edge,
        )
        long_rsi = max(0.0, (profile.range_buy_rsi - rsi_value) / max(profile.range_buy_rsi, 1.0))
        short_rsi = max(
            0.0,
            (rsi_value - profile.range_sell_rsi) / max(100.0 - profile.range_sell_rsi, 1.0),
        )

        long_score = (0.65 * long_edge) + (0.35 * long_rsi)
        short_score = (0.65 * short_edge) + (0.35 * short_rsi)
        score = long_score - short_score

        if not ema_neutral:
            return TradeDecision(
                bias="neutral",
                tradable=False,
                strategy_mode="range_mean_reversion",
                session=session,
                htf_state=current_htf_state,
                score=score,
                threshold=profile.range_threshold,
                atr_ratio=atr_ratio,
                reward_to_risk=min(profile.reward_to_risk, 1.5),
                atr_multiplier=profile.atr_multiplier,
                reason="sideways_not_neutral_enough",
            )

        if score >= profile.range_threshold:
            bias = "bullish"
            tradable = True
        elif score <= -profile.range_threshold:
            bias = "bearish"
            tradable = True
        else:
            bias = "neutral"
            tradable = False

        return TradeDecision(
            bias=bias,
            tradable=tradable,
            strategy_mode="range_mean_reversion",
            session=session,
            htf_state=current_htf_state,
            score=score,
            threshold=profile.range_threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=min(profile.reward_to_risk, 1.5),
            atr_multiplier=profile.atr_multiplier,
            reason=None if tradable else "sideways_signal_not_strong_enough",
        )

    effective_threshold = threshold * profile.trend_threshold_multiplier
    score = float(latest["ltf_score"])

    if score >= effective_threshold:
        bias = "bullish"
    elif score <= -effective_threshold:
        bias = "bearish"
    else:
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="trend_following",
            session=session,
            htf_state=current_htf_state,
            score=score,
            threshold=effective_threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="trend_signal_below_session_threshold",
        )

    if current_htf_state == HTFState.BULLISH.value and bias != "bullish":
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="trend_following",
            session=session,
            htf_state=current_htf_state,
            score=score,
            threshold=effective_threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="htf_bullish_blocks_short",
        )
    if current_htf_state == HTFState.BEARISH.value and bias != "bearish":
        return TradeDecision(
            bias="neutral",
            tradable=False,
            strategy_mode="trend_following",
            session=session,
            htf_state=current_htf_state,
            score=score,
            threshold=effective_threshold,
            atr_ratio=atr_ratio,
            reward_to_risk=profile.reward_to_risk,
            atr_multiplier=profile.atr_multiplier,
            reason="htf_bearish_blocks_long",
        )

    return TradeDecision(
        bias=bias,
        tradable=True,
        strategy_mode="trend_following",
        session=session,
        htf_state=current_htf_state,
        score=score,
        threshold=effective_threshold,
        atr_ratio=atr_ratio,
        reward_to_risk=profile.reward_to_risk,
        atr_multiplier=profile.atr_multiplier,
        reason=None,
    )


def get_htf_policy(state: HTFState | str) -> HTFPolicy:
    normalized_state = HTFState(state)

    if normalized_state is HTFState.BULLISH:
        return HTFPolicy(
            state=normalized_state,
            allow_long=True,
            allow_short=False,
            frequency_divisor=1,
            note="long_only",
        )
    if normalized_state is HTFState.BEARISH:
        return HTFPolicy(
            state=normalized_state,
            allow_long=False,
            allow_short=True,
            frequency_divisor=1,
            note="short_only",
        )
    if normalized_state is HTFState.SIDEWAYS:
        return HTFPolicy(
            state=normalized_state,
            allow_long=True,
            allow_short=True,
            frequency_divisor=1,
            note="range_logic_both_sides",
        )
    return HTFPolicy(
        state=normalized_state,
        allow_long=False,
        allow_short=False,
        frequency_divisor=0,
        note="no_trade_volatile_regime",
    )


def get_latest_htf_signal(
    df: pd.DataFrame,
    *,
    volatile_atr_ratio: float = 0.012,
    bullish_rsi_floor: float = 55.0,
    bearish_rsi_ceiling: float = 45.0,
) -> HTFSignal:
    """Classify the latest higher-timeframe regime into four deterministic states."""
    feature_frame = add_indicators(df)
    latest = feature_frame.iloc[-1]

    atr_ratio = float(latest["atr_14"] / latest["close"]) if latest["close"] else 0.0
    ema_50 = float(latest["ema_50"])
    ema_200 = float(latest["ema_200"])
    close = float(latest["close"])
    rsi_14 = float(latest["rsi_14"])

    trend_score = 0.0
    if close > ema_50 > ema_200 and rsi_14 >= bullish_rsi_floor:
        trend_score = 1.0
        state = HTFState.BULLISH
    elif close < ema_50 < ema_200 and rsi_14 <= bearish_rsi_ceiling:
        trend_score = -1.0
        state = HTFState.BEARISH
    else:
        state = HTFState.SIDEWAYS

    if atr_ratio >= volatile_atr_ratio:
        state = HTFState.VOLATILE

    return HTFSignal(
        state=state,
        atr_ratio=atr_ratio,
        trend_score=trend_score,
        policy=get_htf_policy(state),
    )


def get_htf_state_series(
    df: pd.DataFrame,
    *,
    rule: str = "4H",
    volatile_atr_ratio: float = 0.012,
    bullish_rsi_floor: float = 55.0,
    bearish_rsi_ceiling: float = 45.0,
) -> pd.Series:
    """Project higher-timeframe regime states onto a lower-timeframe candle index."""
    htf_df = resample_ohlc(df, rule)
    htf_states = htf_df.apply(
        lambda _: None,
        axis=1,
    )
    htf_states = htf_df.apply(
        lambda row: get_latest_htf_signal(
            htf_df.loc[: row.name],
            volatile_atr_ratio=volatile_atr_ratio,
            bullish_rsi_floor=bullish_rsi_floor,
            bearish_rsi_ceiling=bearish_rsi_ceiling,
        ).state.value,
        axis=1,
    )

    projected = htf_states.reindex(df.index, method="ffill")
    return projected.fillna(HTFState.SIDEWAYS.value)


def htf_allows_ltf_trade(
    htf_state: HTFState | str,
    ltf_bias: str,
    *,
    signal_index: int = 0,
) -> bool:
    """Return whether a lower-timeframe trade is allowed under the HTF regime."""
    policy = get_htf_policy(htf_state)
    normalized_bias = ltf_bias.lower()

    if normalized_bias == "bullish" and not policy.allow_long:
        return False
    if normalized_bias == "bearish" and not policy.allow_short:
        return False
    if normalized_bias not in {"bullish", "bearish"}:
        return False
    if policy.frequency_divisor <= 0:
        return False
    return signal_index % policy.frequency_divisor == 0


def get_trading_session(timestamp: pd.Timestamp) -> TradingSession:
    """Map a candle timestamp to a broad UTC trading session."""
    hour = timestamp.hour
    if 0 <= hour < 8:
        return TradingSession.ASIA
    if 8 <= hour < 13:
        return TradingSession.LONDON
    if 13 <= hour < 18:
        return TradingSession.NEW_YORK
    return TradingSession.OFF_HOURS


def get_session_profile(session: TradingSession | str) -> SessionProfile:
    normalized_session = TradingSession(session)

    if normalized_session is TradingSession.ASIA:
        return SessionProfile(
            session=normalized_session,
            trend_threshold_multiplier=1.10,
            atr_floor_multiplier=1.05,
            range_edge=0.15,
            range_buy_rsi=35.0,
            range_sell_rsi=65.0,
            range_threshold=0.45,
            reward_to_risk=1.6,
            atr_multiplier=1.4,
            note="quieter_session_more_selective",
        )
    if normalized_session is TradingSession.LONDON:
        return SessionProfile(
            session=normalized_session,
            trend_threshold_multiplier=0.95,
            atr_floor_multiplier=1.00,
            range_edge=0.20,
            range_buy_rsi=40.0,
            range_sell_rsi=60.0,
            range_threshold=0.40,
            reward_to_risk=2.0,
            atr_multiplier=1.5,
            note="most_active_hfm_gold_session",
        )
    if normalized_session is TradingSession.NEW_YORK:
        return SessionProfile(
            session=normalized_session,
            trend_threshold_multiplier=1.00,
            atr_floor_multiplier=1.05,
            range_edge=0.18,
            range_buy_rsi=38.0,
            range_sell_rsi=62.0,
            range_threshold=0.42,
            reward_to_risk=1.9,
            atr_multiplier=1.5,
            note="active_us_session",
        )
    return SessionProfile(
        session=normalized_session,
        trend_threshold_multiplier=1.20,
        atr_floor_multiplier=1.20,
        range_edge=0.10,
        range_buy_rsi=30.0,
        range_sell_rsi=70.0,
        range_threshold=0.50,
        reward_to_risk=1.4,
        atr_multiplier=1.3,
        note="thinner_hours_more_selective",
    )
