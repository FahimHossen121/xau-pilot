from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from trading_bot.htf_ai import HTFAIController, HTFAIEvaluation, HTFAIControllerState, TechnicalHTFSnapshot
from trading_bot.strategies import HTFState, get_latest_htf_signal, resample_ohlc


COLLECTION_COLUMNS = [
    "timestamp",
    "symbol",
    "state",
    "confidence",
    "summary",
    "drivers",
    "invalidates",
    "technical_state",
    "effective_state",
    "last_confirmed_state",
    "trading_enabled",
    "trigger_reason",
    "article_count",
    "stop_reason",
    "news_provider",
    "gemini_model",
    "technical_atr_ratio",
    "technical_trend_score",
    "ai_checked_at",
    "ai_expires_at",
]


def _normalize_utc_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def build_historical_technical_snapshots(
    df: pd.DataFrame,
    *,
    start_time: str | datetime | pd.Timestamp,
    end_time: str | datetime | pd.Timestamp,
    rule: str = "1H",
    volatile_atr_ratio: float = 0.012,
) -> list[TechnicalHTFSnapshot]:
    start_ts = _normalize_utc_timestamp(start_time)
    end_ts = _normalize_utc_timestamp(end_time)
    if start_ts >= end_ts:
        raise ValueError("start_time must be earlier than end_time.")

    htf_df = resample_ohlc(df, rule)
    snapshots: list[TechnicalHTFSnapshot] = []
    for index in range(1, len(htf_df)):
        bar_time = _normalize_utc_timestamp(htf_df.index[index])
        if bar_time < start_ts or bar_time >= end_ts:
            continue

        current_signal = get_latest_htf_signal(
            htf_df.iloc[: index + 1],
            volatile_atr_ratio=volatile_atr_ratio,
        )
        previous_signal = get_latest_htf_signal(
            htf_df.iloc[:index],
            volatile_atr_ratio=volatile_atr_ratio,
        )
        snapshots.append(
            TechnicalHTFSnapshot(
                as_of=bar_time.to_pydatetime(),
                rule=rule,
                current_state=current_signal.state,
                previous_state=previous_signal.state,
                changed=current_signal.state != previous_signal.state,
                atr_ratio=current_signal.atr_ratio,
                trend_score=current_signal.trend_score,
            )
        )
    return snapshots


def _evaluation_to_row(
    evaluation: HTFAIEvaluation,
    *,
    symbol: str,
    news_provider: str,
    gemini_model: str,
) -> dict[str, object]:
    state = evaluation.state
    return {
        "timestamp": evaluation.technical_snapshot.as_of.isoformat(),
        "symbol": symbol,
        "state": state.ai_state.value if state.ai_state else "",
        "confidence": state.ai_confidence,
        "summary": state.ai_summary or "",
        "drivers": " | ".join(state.drivers),
        "invalidates": " | ".join(state.invalidates),
        "technical_state": evaluation.technical_snapshot.current_state.value,
        "effective_state": state.effective_state.value if state.effective_state else "",
        "last_confirmed_state": (
            state.last_confirmed_state.value if state.last_confirmed_state else ""
        ),
        "trading_enabled": state.trading_enabled,
        "trigger_reason": evaluation.trigger_reason,
        "article_count": evaluation.article_count,
        "stop_reason": state.stop_reason or "",
        "news_provider": news_provider,
        "gemini_model": gemini_model,
        "technical_atr_ratio": evaluation.technical_snapshot.atr_ratio,
        "technical_trend_score": evaluation.technical_snapshot.trend_score,
        "ai_checked_at": state.ai_checked_at.isoformat() if state.ai_checked_at else "",
        "ai_expires_at": state.ai_expires_at.isoformat() if state.ai_expires_at else "",
    }


def _split_pipe_field(raw: object) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _parse_optional_htf_state(raw: object) -> HTFState | None:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    return HTFState(value)


def load_collection_resume_state(
    path: str | Path,
) -> tuple[HTFAIControllerState | None, pd.Timestamp | None, int]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None, None, 0

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None, None, 0

    row = frame.iloc[-1]
    timestamp = _normalize_utc_timestamp(row["timestamp"])
    ai_checked_at = row.get("ai_checked_at")
    ai_expires_at = row.get("ai_expires_at")
    state = HTFAIControllerState(
        last_confirmed_state=_parse_optional_htf_state(row.get("last_confirmed_state")),
        effective_state=_parse_optional_htf_state(row.get("effective_state")),
        technical_state=_parse_optional_htf_state(row.get("technical_state")),
        ai_state=_parse_optional_htf_state(row.get("state")),
        trading_enabled=bool(row.get("trading_enabled", False)),
        ai_checked_at=(
            _normalize_utc_timestamp(ai_checked_at).to_pydatetime()
            if str(ai_checked_at or "").strip()
            else None
        ),
        ai_expires_at=(
            _normalize_utc_timestamp(ai_expires_at).to_pydatetime()
            if str(ai_expires_at or "").strip()
            else None
        ),
        ai_confidence=(
            float(row["confidence"])
            if pd.notna(row.get("confidence"))
            else None
        ),
        ai_summary=str(row.get("summary") or "").strip() or None,
        drivers=_split_pipe_field(row.get("drivers")),
        invalidates=_split_pipe_field(row.get("invalidates")),
        last_trigger_reason=str(row.get("trigger_reason") or "").strip() or None,
        stop_reason=str(row.get("stop_reason") or "").strip() or None,
    )
    return state, timestamp, len(frame)


def iter_historical_ai_replay_rows(
    *,
    symbol: str,
    snapshots: Iterable[TechnicalHTFSnapshot],
    controller: HTFAIController,
    news_provider: str,
    gemini_model: str,
    allow_expiry_refresh: bool = False,
    seconds_between_ai_calls: float = 0.0,
    initial_state: HTFAIControllerState | None = None,
    skip_until: str | datetime | pd.Timestamp | None = None,
    max_ai_calls: int | None = None,
):
    snapshot_list = list(snapshots)
    state = initial_state or HTFAIControllerState()
    skip_until_ts = _normalize_utc_timestamp(skip_until) if skip_until is not None else None
    ai_call_count = 0
    first_iteration = True
    for index, snapshot in enumerate(snapshot_list):
        snapshot_ts = _normalize_utc_timestamp(snapshot.as_of)
        if skip_until_ts is not None and snapshot_ts <= skip_until_ts:
            continue

        evaluation = controller.evaluate(
            symbol=symbol,
            technical_snapshot=snapshot,
            state=state,
            now=snapshot.as_of,
            force_refresh=first_iteration and state.ai_state is None,
            allow_expiry_refresh=allow_expiry_refresh,
        )
        first_iteration = False
        state = evaluation.state
        if evaluation.ai_called:
            ai_call_count += 1
            row = _evaluation_to_row(
                evaluation,
                symbol=symbol,
                news_provider=news_provider,
                gemini_model=gemini_model,
            )
            yield row, state
            if max_ai_calls is not None and ai_call_count >= max_ai_calls:
                return
            if seconds_between_ai_calls > 0 and index < len(snapshot_list) - 1:
                time.sleep(seconds_between_ai_calls)


def collect_historical_ai_replay(
    *,
    symbol: str,
    snapshots: Iterable[TechnicalHTFSnapshot],
    controller: HTFAIController,
    news_provider: str,
    gemini_model: str,
    allow_expiry_refresh: bool = False,
    seconds_between_ai_calls: float = 0.0,
    initial_state: HTFAIControllerState | None = None,
    skip_until: str | datetime | pd.Timestamp | None = None,
    max_ai_calls: int | None = None,
) -> pd.DataFrame:
    rows = [
        row
        for row, _ in iter_historical_ai_replay_rows(
            symbol=symbol,
            snapshots=snapshots,
            controller=controller,
            news_provider=news_provider,
            gemini_model=gemini_model,
            allow_expiry_refresh=allow_expiry_refresh,
            seconds_between_ai_calls=seconds_between_ai_calls,
            initial_state=initial_state,
            skip_until=skip_until,
            max_ai_calls=max_ai_calls,
        )
    ]
    return pd.DataFrame(rows, columns=COLLECTION_COLUMNS)
