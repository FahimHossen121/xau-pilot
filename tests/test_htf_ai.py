from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.config import Settings
from trading_bot.htf_ai import (
    AIMacroAssessment,
    HTFAIController,
    HTFAIControllerState,
    HTFAITrigger,
    NewsArticle,
    TechnicalHTFSnapshot,
    build_technical_htf_snapshot,
    build_live_controller,
    load_htf_ai_state,
    save_htf_ai_state,
)
from trading_bot.strategies import HTFState


class FakeNewsProvider:
    def __init__(self, articles: list[NewsArticle]) -> None:
        self.articles = articles
        self.calls = 0

    def gather_context(self, **kwargs) -> list[NewsArticle]:
        self.calls += 1
        return self.articles


class FakeMacroAnalyzer:
    def __init__(self, assessment: AIMacroAssessment) -> None:
        self.assessment = assessment
        self.calls = 0

    def analyze(self, **kwargs) -> AIMacroAssessment:
        self.calls += 1
        return self.assessment


def test_build_technical_htf_snapshot_detects_trend() -> None:
    rows = 400
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
        },
        index=pd.date_range("2026-03-01", periods=rows, freq="5min", tz="UTC"),
    )

    snapshot = build_technical_htf_snapshot(df, rule="1H", volatile_atr_ratio=0.02)

    assert snapshot.current_state is HTFState.BULLISH
    assert snapshot.previous_state is not None
    assert snapshot.as_of.tzinfo is not None


def test_controller_startup_match_enables_trading() -> None:
    provider = FakeNewsProvider(
        [NewsArticle("Gold supported", "https://example.com/1", "example", None, "Snippet")]
    )
    analyzer = FakeMacroAnalyzer(
        AIMacroAssessment(
            state=HTFState.BULLISH,
            confidence=0.81,
            summary="Macro matches bullish trend.",
            drivers=("weaker dollar",),
            invalidates=("hawkish Fed repricing",),
            expires_in_hours=1,
        )
    )
    controller = HTFAIController(
        news_provider=provider,
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    snapshot = TechnicalHTFSnapshot(
        as_of=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
        rule="1H",
        current_state=HTFState.BULLISH,
        previous_state=HTFState.BULLISH,
        changed=False,
        atr_ratio=0.003,
        trend_score=1.0,
    )

    evaluation = controller.evaluate(symbol="XAUUSD", technical_snapshot=snapshot)

    assert evaluation.trigger_reason == HTFAITrigger.STARTUP
    assert evaluation.ai_called is True
    assert evaluation.state.trading_enabled is True
    assert evaluation.state.effective_state is HTFState.BULLISH
    assert provider.calls == 1
    assert analyzer.calls == 1


def test_controller_expiry_mismatch_stops_trading() -> None:
    provider = FakeNewsProvider([])
    analyzer = FakeMacroAnalyzer(
        AIMacroAssessment(
            state=HTFState.BEARISH,
            confidence=0.67,
            summary="Macro disagrees.",
            drivers=("rising yields",),
            invalidates=("risk-off bid",),
            expires_in_hours=1,
        )
    )
    controller = HTFAIController(
        news_provider=provider,
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    prior_state = HTFAIControllerState(
        last_confirmed_state=HTFState.BULLISH,
        effective_state=HTFState.BULLISH,
        technical_state=HTFState.BULLISH,
        ai_state=HTFState.BULLISH,
        trading_enabled=True,
        ai_checked_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
        ai_expires_at=datetime(2026, 3, 25, 11, 0, tzinfo=timezone.utc),
        ai_confidence=0.8,
        ai_summary="Still bullish.",
    )
    snapshot = TechnicalHTFSnapshot(
        as_of=datetime(2026, 3, 25, 11, 0, tzinfo=timezone.utc),
        rule="1H",
        current_state=HTFState.BULLISH,
        previous_state=HTFState.BULLISH,
        changed=False,
        atr_ratio=0.003,
        trend_score=1.0,
    )

    evaluation = controller.evaluate(
        symbol="XAUUSD",
        technical_snapshot=snapshot,
        state=prior_state,
        now=snapshot.as_of,
    )

    assert evaluation.trigger_reason == HTFAITrigger.EXPIRY
    assert evaluation.state.trading_enabled is False
    assert evaluation.state.stop_reason == "technical_ai_mismatch"
    assert evaluation.state.effective_state is None


def test_controller_technical_shift_rechecks_before_expiry() -> None:
    provider = FakeNewsProvider([])
    analyzer = FakeMacroAnalyzer(
        AIMacroAssessment(
            state=HTFState.BEARISH,
            confidence=0.74,
            summary="Macro confirms the bearish shift.",
            drivers=("stronger dollar",),
            invalidates=("renewed safe-haven flows",),
            expires_in_hours=1,
        )
    )
    controller = HTFAIController(
        news_provider=provider,
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    prior_state = HTFAIControllerState(
        last_confirmed_state=HTFState.BULLISH,
        effective_state=HTFState.BULLISH,
        technical_state=HTFState.BULLISH,
        ai_state=HTFState.BULLISH,
        trading_enabled=True,
        ai_checked_at=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
        ai_expires_at=datetime(2026, 3, 25, 11, 0, tzinfo=timezone.utc),
    )
    snapshot = TechnicalHTFSnapshot(
        as_of=datetime(2026, 3, 25, 10, 30, tzinfo=timezone.utc),
        rule="1H",
        current_state=HTFState.BEARISH,
        previous_state=HTFState.BULLISH,
        changed=True,
        atr_ratio=0.004,
        trend_score=-1.0,
    )

    evaluation = controller.evaluate(
        symbol="XAUUSD",
        technical_snapshot=snapshot,
        state=prior_state,
        now=snapshot.as_of,
    )

    assert evaluation.trigger_reason == HTFAITrigger.TECHNICAL_SHIFT
    assert evaluation.state.trading_enabled is True
    assert evaluation.state.effective_state is HTFState.BEARISH


def test_controller_technical_volatile_stops_without_ai_call() -> None:
    provider = FakeNewsProvider([])
    analyzer = FakeMacroAnalyzer(
        AIMacroAssessment(
            state=HTFState.BULLISH,
            confidence=0.9,
            summary="Unused",
            drivers=(),
            invalidates=(),
            expires_in_hours=1,
        )
    )
    controller = HTFAIController(
        news_provider=provider,
        macro_analyzer=analyzer,
        refresh_hours=1,
    )
    snapshot = TechnicalHTFSnapshot(
        as_of=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
        rule="1H",
        current_state=HTFState.VOLATILE,
        previous_state=HTFState.BULLISH,
        changed=True,
        atr_ratio=0.02,
        trend_score=0.0,
    )

    evaluation = controller.evaluate(symbol="XAUUSD", technical_snapshot=snapshot)

    assert evaluation.trigger_reason == HTFAITrigger.TECHNICAL_VOLATILE
    assert evaluation.ai_called is False
    assert evaluation.state.trading_enabled is False
    assert provider.calls == 0
    assert analyzer.calls == 0


def test_htf_ai_state_roundtrip() -> None:
    path = Path("tests/.htf_state.test.json")
    state = HTFAIControllerState(
        last_confirmed_state=HTFState.BULLISH,
        effective_state=HTFState.BULLISH,
        technical_state=HTFState.BULLISH,
        ai_state=HTFState.BULLISH,
        trading_enabled=True,
        ai_checked_at=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
        ai_expires_at=datetime(2026, 3, 25, 13, 0, tzinfo=timezone.utc),
        ai_confidence=0.7,
        ai_summary="Bullish remains intact.",
        drivers=("risk-off",),
        invalidates=("rising yields",),
        last_trigger_reason=HTFAITrigger.EXPIRY,
        stop_reason=None,
    )

    try:
        save_htf_ai_state(path, state)
        loaded = load_htf_ai_state(path)

        assert loaded.last_confirmed_state is HTFState.BULLISH
        assert loaded.effective_state is HTFState.BULLISH
        assert loaded.ai_state is HTFState.BULLISH
        assert loaded.trading_enabled is True
        assert loaded.ai_summary == "Bullish remains intact."
    finally:
        path.unlink(missing_ok=True)


def test_build_live_controller_allows_free_none_provider() -> None:
    settings = Settings(
        app_mode="paper",
        log_level="INFO",
        timezone="UTC",
        symbol="XAUUSD",
        max_risk_per_trade=0.01,
        max_daily_loss=0.03,
        mt5_login=None,
        mt5_password=None,
        mt5_server=None,
        news_provider="none",
        rss_feed_urls=(),
        brave_api_key=None,
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-1.5-pro",
        ai_htf_refresh_hours=1,
        brave_news_freshness="pd",
        brave_news_results_per_query=5,
        enable_trading=False,
    )

    controller = build_live_controller(settings)

    assert isinstance(controller, HTFAIController)
