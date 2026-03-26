from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.strategies import HTFState, get_latest_htf_signal, resample_ohlc


REQUIRED_AI_REPLAY_COLUMNS = {"timestamp", "state"}


def load_ai_state_history(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"AI replay CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_AI_REPLAY_COLUMNS.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"AI replay CSV is missing required columns: {missing_columns}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["state"] = frame["state"].astype(str).str.strip().str.lower()
    invalid_states = sorted(
        {
            value
            for value in frame["state"].dropna().unique().tolist()
            if value not in {state.value for state in HTFState}
        }
    )
    if invalid_states:
        invalid = ", ".join(invalid_states)
        raise ValueError(f"AI replay CSV contains invalid states: {invalid}")

    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    frame.set_index("timestamp", inplace=True)
    return frame


def project_ai_state_series(
    lower_timeframe_index: pd.DatetimeIndex,
    ai_history: pd.DataFrame,
) -> pd.Series:
    if not isinstance(lower_timeframe_index, pd.DatetimeIndex):
        raise ValueError("lower_timeframe_index must be a DatetimeIndex.")
    if "state" not in ai_history.columns:
        raise ValueError("ai_history must contain a 'state' column.")

    normalized_index = pd.DatetimeIndex(lower_timeframe_index)
    if normalized_index.tz is None:
        normalized_index = normalized_index.tz_localize("UTC")
    else:
        normalized_index = normalized_index.tz_convert("UTC")

    ai_states = ai_history["state"].copy()
    if ai_states.index.tz is None:
        ai_states.index = ai_states.index.tz_localize("UTC")
    else:
        ai_states.index = ai_states.index.tz_convert("UTC")
    projected = ai_states.reindex(normalized_index, method="ffill")
    return projected


def build_effective_htf_series(
    *,
    technical_states: pd.Series,
    ai_states: pd.Series,
) -> pd.Series:
    if len(technical_states) != len(ai_states):
        raise ValueError("technical_states and ai_states must have the same length.")

    effective = pd.Series(index=technical_states.index, dtype="object")
    for timestamp, technical_state, ai_state in zip(
        technical_states.index,
        technical_states.tolist(),
        ai_states.tolist(),
        strict=False,
    ):
        if pd.isna(technical_state) or pd.isna(ai_state):
            effective.loc[timestamp] = HTFState.VOLATILE.value
            continue
        if technical_state == HTFState.VOLATILE.value:
            effective.loc[timestamp] = HTFState.VOLATILE.value
            continue
        if technical_state == ai_state:
            effective.loc[timestamp] = technical_state
            continue
        effective.loc[timestamp] = HTFState.VOLATILE.value
    return effective


def build_technical_seed_ai_history(
    df: pd.DataFrame,
    *,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    rule: str = "1H",
    volatile_atr_ratio: float = 0.012,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if start_ts >= end_ts:
        raise ValueError("start_time must be earlier than end_time.")

    htf_df = resample_ohlc(df, rule)
    rows: list[dict[str, object]] = []
    for index in range(1, len(htf_df)):
        bar_time = pd.Timestamp(htf_df.index[index])
        if bar_time.tzinfo is None:
            bar_time = bar_time.tz_localize("UTC")
        else:
            bar_time = bar_time.tz_convert("UTC")
        if bar_time < start_ts or bar_time >= end_ts:
            continue
        signal = get_latest_htf_signal(
            htf_df.iloc[: index + 1],
            volatile_atr_ratio=volatile_atr_ratio,
        )
        if signal.state is HTFState.VOLATILE:
            confidence = 0.85
        elif signal.state is HTFState.SIDEWAYS:
            confidence = 0.55
        else:
            confidence = 0.65
        rows.append(
            {
                "timestamp": bar_time,
                "state": signal.state.value,
                "confidence": confidence,
                "summary": "Technical seed generated from H1 EMA/RSI/ATR regime classification.",
                "seed_source": "technical_only",
            }
        )

    return pd.DataFrame(rows, columns=["timestamp", "state", "confidence", "summary", "seed_source"])
