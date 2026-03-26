from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import run_ltf_backtest
from trading_bot.htf_ai_replay import (
    build_technical_seed_ai_history,
    build_effective_htf_series,
    load_ai_state_history,
    project_ai_state_series,
)
from trading_bot.strategies import HTFState, get_htf_state_series


def test_load_ai_state_history_normalizes_and_sorts() -> None:
    path = Path("tests/.ai_replay.test.csv")
    path.write_text(
        "\n".join(
            [
                "timestamp,state,confidence",
                "2026-03-01T02:00:00Z,bearish,0.6",
                "2026-03-01T01:00:00Z,bullish,0.7",
            ]
        ),
        encoding="utf-8",
    )

    try:
        frame = load_ai_state_history(path)
    finally:
        path.unlink(missing_ok=True)

    assert list(frame["state"]) == ["bullish", "bearish"]
    assert frame.index.tz is not None


def test_build_effective_htf_series_requires_match() -> None:
    index = pd.date_range("2026-03-01", periods=3, freq="1h", tz="UTC")
    technical = pd.Series(
        [HTFState.BULLISH.value, HTFState.BEARISH.value, HTFState.VOLATILE.value],
        index=index,
    )
    ai = pd.Series(
        [HTFState.BULLISH.value, HTFState.BULLISH.value, HTFState.BEARISH.value],
        index=index,
    )

    effective = build_effective_htf_series(technical_states=technical, ai_states=ai)

    assert list(effective) == [
        HTFState.BULLISH.value,
        HTFState.VOLATILE.value,
        HTFState.VOLATILE.value,
    ]


def test_run_ltf_backtest_with_ai_replay_blocks_mismatched_trades() -> None:
    rows = 500
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 0.5,
            "close": close,
        },
        index=pd.date_range("2026-03-01", periods=rows, freq="5min", tz="UTC"),
    )

    replay_path = Path("tests/.ai_replay_mismatch.test.csv")
    replay_path.write_text(
        "\n".join(
            [
                "timestamp,state",
                "2026-03-01T00:00:00Z,bearish",
                "2026-03-01T06:00:00Z,bearish",
                "2026-03-01T12:00:00Z,bearish",
            ]
        ),
        encoding="utf-8",
    )

    try:
        result = run_ltf_backtest(
            df,
            use_htf_filter=True,
            htf_rule="1H",
            ai_htf_replay_path=replay_path,
        )
    finally:
        replay_path.unlink(missing_ok=True)

    assert result.used_ai_htf_replay is True
    assert result.trade_count == 0


def test_projected_ai_replay_allows_matching_states() -> None:
    rows = 500
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 0.5,
            "close": close,
        },
        index=pd.date_range("2026-03-01", periods=rows, freq="5min", tz="UTC"),
    )

    technical_states = get_htf_state_series(df, rule="1H", volatile_atr_ratio=0.05)
    replay_index = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    ai_history = pd.DataFrame({"state": [HTFState.BULLISH.value] * len(replay_index)}, index=replay_index)
    projected = project_ai_state_series(df.index, ai_history)
    effective = build_effective_htf_series(technical_states=technical_states, ai_states=projected)

    assert (effective == HTFState.BULLISH.value).any()


def test_build_technical_seed_ai_history_outputs_expected_columns() -> None:
    rows = 800
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=pd.date_range("2026-03-01", periods=rows, freq="5min", tz="UTC"),
    )

    seed_df = build_technical_seed_ai_history(
        df,
        start_time="2026-03-02T00:00:00Z",
        end_time="2026-03-03T00:00:00Z",
        rule="1H",
        volatile_atr_ratio=0.05,
    )

    assert not seed_df.empty
    assert list(seed_df.columns) == [
        "timestamp",
        "state",
        "confidence",
        "summary",
        "seed_source",
    ]
    assert seed_df["seed_source"].eq("technical_only").all()
