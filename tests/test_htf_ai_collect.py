from datetime import datetime, timezone
from pathlib import Path

from trading_bot.htf_ai import (
    AIMacroAssessment,
    HTFAIController,
    NewsArticle,
    TechnicalHTFSnapshot,
)
from trading_bot.htf_ai_collect import collect_historical_ai_replay
from trading_bot.htf_ai_collect import load_collection_resume_state
from trading_bot.strategies import HTFState


class FakeNewsProvider:
    def gather_context(self, **kwargs) -> list[NewsArticle]:
        return []


class EchoMacroAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, **kwargs) -> AIMacroAssessment:
        self.calls += 1
        snapshot = kwargs["technical_snapshot"]
        return AIMacroAssessment(
            state=snapshot.current_state,
            confidence=0.75,
            summary=f"Echoed {snapshot.current_state.value} state.",
            drivers=("echo",),
            invalidates=(),
            expires_in_hours=1,
        )


def _snapshot(
    hour: int,
    *,
    current: HTFState,
    previous: HTFState,
    changed: bool,
) -> TechnicalHTFSnapshot:
    return TechnicalHTFSnapshot(
        as_of=datetime(2026, 3, 1, hour, 0, tzinfo=timezone.utc),
        rule="1H",
        current_state=current,
        previous_state=previous,
        changed=changed,
        atr_ratio=0.003,
        trend_score=1.0 if current is HTFState.BULLISH else -1.0,
    )


def test_collect_historical_ai_replay_shift_only_calls_startup_and_shifts() -> None:
    analyzer = EchoMacroAnalyzer()
    controller = HTFAIController(
        news_provider=FakeNewsProvider(),
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    snapshots = [
        _snapshot(0, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
        _snapshot(1, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
        _snapshot(2, current=HTFState.BEARISH, previous=HTFState.BULLISH, changed=True),
        _snapshot(3, current=HTFState.BEARISH, previous=HTFState.BEARISH, changed=False),
    ]

    replay_df = collect_historical_ai_replay(
        symbol="XAUUSD",
        snapshots=snapshots,
        controller=controller,
        news_provider="none",
        gemini_model="gemini-1.5-pro",
        allow_expiry_refresh=False,
    )

    assert analyzer.calls == 2
    assert replay_df["trigger_reason"].tolist() == ["startup", "technical_shift"]
    assert replay_df["state"].tolist() == ["bullish", "bearish"]


def test_collect_historical_ai_replay_with_expiry_refresh_calls_more_often() -> None:
    analyzer = EchoMacroAnalyzer()
    controller = HTFAIController(
        news_provider=FakeNewsProvider(),
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    snapshots = [
        _snapshot(0, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
        _snapshot(1, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
        _snapshot(2, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
    ]

    replay_df = collect_historical_ai_replay(
        symbol="XAUUSD",
        snapshots=snapshots,
        controller=controller,
        news_provider="none",
        gemini_model="gemini-1.5-pro",
        allow_expiry_refresh=True,
    )

    assert analyzer.calls == 3
    assert replay_df["trigger_reason"].tolist() == ["startup", "expiry", "expiry"]


def test_load_collection_resume_state_restores_last_ai_state() -> None:
    analyzer = EchoMacroAnalyzer()
    controller = HTFAIController(
        news_provider=FakeNewsProvider(),
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    snapshots = [
        _snapshot(0, current=HTFState.BULLISH, previous=HTFState.BULLISH, changed=False),
        _snapshot(1, current=HTFState.BEARISH, previous=HTFState.BULLISH, changed=True),
    ]
    replay_df = collect_historical_ai_replay(
        symbol="XAUUSD",
        snapshots=snapshots,
        controller=controller,
        news_provider="none",
        gemini_model="gemini-2.5-flash",
        allow_expiry_refresh=False,
    )

    path = Path("tests/.ai_collect_resume.test.csv")
    replay_df.to_csv(path, index=False)
    try:
        state, timestamp, rows = load_collection_resume_state(path)
    finally:
        path.unlink(missing_ok=True)

    assert rows == 2
    assert timestamp is not None
    assert state is not None
    assert state.ai_state is HTFState.BEARISH
    assert state.last_confirmed_state is HTFState.BEARISH
