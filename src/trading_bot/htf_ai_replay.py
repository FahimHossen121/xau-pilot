from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_bot.strategies import HTFState


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

    ai_states = ai_history["state"].copy()
    if ai_states.index.tz is None:
        ai_states.index = ai_states.index.tz_localize("UTC")
    projected = ai_states.reindex(lower_timeframe_index, method="ffill")
    return projected


def build_effective_htf_series(
    *,
    technical_states: pd.Series,
    ai_states: pd.Series,
) -> pd.Series:
    if len(technical_states) != len(ai_states):
        raise ValueError("technical_states and ai_states must have the same length.")

    effective = pd.Series(index=technical_states.index, dtype="object")
    for timestamp in technical_states.index:
        technical_state = technical_states.loc[timestamp]
        ai_state = ai_states.loc[timestamp]
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
