import numpy as np
import pandas as pd

from trading_bot.backtest import format_backtest_summary, run_ltf_backtest


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


def test_run_ltf_backtest_transaction_costs_reduce_results() -> None:
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

    zero_cost = run_ltf_backtest(df, spread=0.0, slippage=0.0)
    with_cost = run_ltf_backtest(df, spread=0.5, slippage=0.1)

    assert with_cost.final_balance < zero_cost.final_balance
    assert sum(trade.transaction_cost for trade in with_cost.trades) > 0


def test_run_ltf_backtest_with_htf_filter_blocks_volatile_regime() -> None:
    rows = 320
    close = np.linspace(100.0, 180.0, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 8.0,
            "low": close - 8.0,
            "close": close,
        },
        index=pd.date_range("2025-01-01", periods=rows, freq="15min"),
    )

    no_filter = run_ltf_backtest(df, use_htf_filter=False)
    with_filter = run_ltf_backtest(df, use_htf_filter=True, htf_rule="4H", htf_volatile_atr_ratio=0.01)

    assert no_filter.trade_count > 0
    assert with_filter.trade_count == 0


def test_format_backtest_summary_includes_core_metrics() -> None:
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
    summary = format_backtest_summary(result)

    assert "Backtest Summary" in summary
    assert "Final balance:" in summary
    assert "Trades:" in summary
    assert "Profit factor:" in summary
    assert "Max drawdown:" in summary
    assert "HTF filter enabled:" in summary
