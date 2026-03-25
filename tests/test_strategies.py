import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies import (
    HTFState,
    add_indicators,
    add_ltf_features,
    get_htf_policy,
    get_latest_htf_signal,
    get_latest_ltf_signal,
    htf_allows_ltf_trade,
)


def test_add_indicators_adds_expected_columns() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 125.0, rows))
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )

    result = add_indicators(df)

    assert {"ema_50", "ema_200", "rsi_14", "atr_14"}.issubset(result.columns)
    assert result["ema_50"].notna().all()
    assert result["ema_200"].notna().all()
    assert result["rsi_14"].between(0, 100).all()
    assert (result["atr_14"] >= 0).all()


def test_add_indicators_requires_ohlc_columns() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="required columns"):
        add_indicators(df)


def test_add_ltf_features_adds_signal_columns() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 130.0, rows))
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )

    result = add_ltf_features(df)

    assert {"ema_signal", "rsi_signal", "structure_signal", "atr_ratio", "ltf_score"}.issubset(
        result.columns
    )
    assert result["rsi_signal"].between(-1, 1).all()
    assert set(result["structure_signal"].unique()).issubset({-1.0, 0.0, 1.0})


def test_get_latest_ltf_signal_returns_bullish_for_strong_uptrend() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 150.0, rows))
    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )

    signal = get_latest_ltf_signal(df)

    assert signal.tradable is True
    assert signal.bias == "bullish"
    assert signal.score >= signal.threshold
    assert signal.component_scores["ema"] == 1.0


def test_get_latest_ltf_signal_blocks_low_volatility_market() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 100.5, rows))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        }
    )

    signal = get_latest_ltf_signal(df, atr_floor_ratio=0.01)

    assert signal.tradable is False
    assert signal.bias == "neutral"
    assert signal.reason == "atr_below_floor"


def test_get_htf_policy_matches_four_state_rules() -> None:
    bullish = get_htf_policy(HTFState.BULLISH)
    bearish = get_htf_policy(HTFState.BEARISH)
    sideways = get_htf_policy(HTFState.SIDEWAYS)
    volatile = get_htf_policy(HTFState.VOLATILE)

    assert (bullish.allow_long, bullish.allow_short, bullish.frequency_divisor) == (True, False, 1)
    assert (bearish.allow_long, bearish.allow_short, bearish.frequency_divisor) == (False, True, 1)
    assert (sideways.allow_long, sideways.allow_short, sideways.frequency_divisor) == (True, True, 3)
    assert (volatile.allow_long, volatile.allow_short, volatile.frequency_divisor) == (False, False, 0)


def test_get_latest_htf_signal_detects_bullish_state() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 150.0, rows))
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
        }
    )

    signal = get_latest_htf_signal(df)

    assert signal.state is HTFState.BULLISH
    assert signal.policy.allow_long is True
    assert signal.policy.allow_short is False


def test_get_latest_htf_signal_detects_volatile_state() -> None:
    rows = 250
    close = pd.Series(np.linspace(100.0, 120.0, rows))
    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
        }
    )

    signal = get_latest_htf_signal(df, volatile_atr_ratio=0.01)

    assert signal.state is HTFState.VOLATILE
    assert signal.policy.allow_long is False
    assert signal.policy.allow_short is False


def test_htf_allows_ltf_trade_respects_sideways_frequency_control() -> None:
    assert htf_allows_ltf_trade(HTFState.SIDEWAYS, "bullish", signal_index=0) is True
    assert htf_allows_ltf_trade(HTFState.SIDEWAYS, "bearish", signal_index=1) is False
    assert htf_allows_ltf_trade(HTFState.BULLISH, "bearish", signal_index=0) is False
    assert htf_allows_ltf_trade(HTFState.VOLATILE, "bullish", signal_index=0) is False
