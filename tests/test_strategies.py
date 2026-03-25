import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies import add_indicators, add_ltf_features, get_latest_ltf_signal


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
