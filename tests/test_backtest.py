import numpy as np
import pandas as pd

from trading_bot.backtest import run_ltf_backtest


def test_run_ltf_backtest_profitable_on_clear_uptrend() -> None:
    rows = 320
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 3.0,
            "low": close - 1.0,
            "close": close,
        },
        index=pd.date_range("2025-01-01", periods=rows, freq="15min"),
    )

    result = run_ltf_backtest(df)

    assert result.trade_count > 0
    assert result.final_balance > result.initial_balance
    assert result.win_count >= 1


def test_run_ltf_backtest_skips_flat_market_when_atr_filter_is_high() -> None:
    rows = 320
    close = np.linspace(100.0, 100.3, rows)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        },
        index=pd.date_range("2025-01-01", periods=rows, freq="15min"),
    )

    result = run_ltf_backtest(df, atr_floor_ratio=0.01)

    assert result.trade_count == 0
    assert result.final_balance == result.initial_balance
