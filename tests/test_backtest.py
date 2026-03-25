import numpy as np
import pandas as pd

from trading_bot.backtest import (
    backtest_result_to_row,
    backtest_trades_to_frame,
    format_backtest_summary,
    run_ltf_backtest,
)


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


def test_run_ltf_backtest_includes_session_and_strategy_mode_on_trades() -> None:
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
    assert result.trades[0].session is not None
    assert result.trades[0].strategy_mode in {"trend_following", "range_mean_reversion"}


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


def test_backtest_export_helpers_build_summary_and_trade_rows() -> None:
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

    result = run_ltf_backtest(df, use_htf_filter=True, htf_rule="4H")
    summary_row = backtest_result_to_row(
        result,
        scenario_name="demo",
        symbol="XAUUSD",
        timeframe="M15",
        candle_count=rows,
        risk_fraction=0.01,
        spread=0.30,
        slippage=0.05,
    )
    trades_df = backtest_trades_to_frame(
        result,
        scenario_name="demo",
        symbol="XAUUSD",
        timeframe="M15",
    )

    assert summary_row["scenario_name"] == "demo"
    assert summary_row["symbol"] == "XAUUSD"
    assert "profit_factor" in summary_row
    assert "max_drawdown_pct" in summary_row
    assert "scenario_name" in trades_df.columns
    assert "session" in trades_df.columns
    assert "strategy_mode" in trades_df.columns
    assert "transaction_cost" in trades_df.columns
