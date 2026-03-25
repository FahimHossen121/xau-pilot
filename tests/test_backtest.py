import numpy as np
import pandas as pd
from trading_bot.risk import TradePlan
from trading_bot.strategies import TradeDecision, TradingSession

from trading_bot.backtest import (
    backtest_result_to_row,
    backtest_session_stats_to_frame,
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
    assert "Cooldown events:" in summary
    assert "Daily lockout days:" in summary
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
    sessions_df = backtest_session_stats_to_frame(
        result,
        scenario_name="demo",
        symbol="XAUUSD",
        timeframe="M15",
    )

    assert summary_row["scenario_name"] == "demo"
    assert summary_row["symbol"] == "XAUUSD"
    assert "profit_factor" in summary_row
    assert "max_drawdown_pct" in summary_row
    assert "cooldown_events" in summary_row
    assert "daily_loss_lockout_days" in summary_row
    assert summary_row["one_open_position_rule"] is True
    assert "scenario_name" in trades_df.columns
    assert "session" in trades_df.columns
    assert "strategy_mode" in trades_df.columns
    assert "transaction_cost" in trades_df.columns
    assert "session" in sessions_df.columns
    assert "strategy_mode" in sessions_df.columns
    assert "profit_factor" in sessions_df.columns


def test_run_ltf_backtest_daily_loss_lockout_stops_new_trades(monkeypatch) -> None:
    rows = 220
    close = np.full(rows, 100.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 2.0,
            "close": close - 0.5,
        },
        index=pd.date_range("2025-01-03 00:00:00", periods=rows, freq="15min"),
    )

    def fake_decision(*args, **kwargs) -> TradeDecision:
        return TradeDecision(
            bias="bullish",
            tradable=True,
            strategy_mode="trend_following",
            session=TradingSession.LONDON,
            htf_state=None,
            score=1.0,
            threshold=0.4,
            atr_ratio=0.01,
            reward_to_risk=2.0,
            atr_multiplier=1.5,
        )

    def fake_trade_plan(*, side, entry_price, atr_value, account_balance, risk_fraction, atr_multiplier, reward_to_risk):
        risk_amount = account_balance * risk_fraction
        return TradePlan(
            side=side,
            entry_price=entry_price,
            stop_loss=entry_price - 1.0,
            take_profit_1=entry_price + 1.0,
            take_profit_2=entry_price + 2.0,
            stop_distance=1.0,
            risk_amount=risk_amount,
            position_size=risk_amount,
            reward_to_risk=reward_to_risk,
        )

    monkeypatch.setattr("trading_bot.backtest.get_trade_decision", fake_decision)
    monkeypatch.setattr("trading_bot.backtest.build_trade_plan", fake_trade_plan)

    result = run_ltf_backtest(
        df,
        max_daily_loss_fraction=0.019,
        cooldown_bars_after_loss=0,
        loss_streak_for_cooldown=1,
    )

    assert result.trade_count == 2
    assert result.daily_loss_lockout_days == 1


def test_run_ltf_backtest_cooldown_reduces_reentry_after_losses(monkeypatch) -> None:
    rows = 220
    close = np.full(rows, 100.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 2.0,
            "close": close - 0.5,
        },
        index=pd.date_range("2025-01-03 00:00:00", periods=rows, freq="15min"),
    )

    def fake_decision(*args, **kwargs) -> TradeDecision:
        return TradeDecision(
            bias="bullish",
            tradable=True,
            strategy_mode="trend_following",
            session=TradingSession.LONDON,
            htf_state=None,
            score=1.0,
            threshold=0.4,
            atr_ratio=0.01,
            reward_to_risk=2.0,
            atr_multiplier=1.5,
        )

    def fake_trade_plan(*, side, entry_price, atr_value, account_balance, risk_fraction, atr_multiplier, reward_to_risk):
        risk_amount = account_balance * risk_fraction
        return TradePlan(
            side=side,
            entry_price=entry_price,
            stop_loss=entry_price - 1.0,
            take_profit_1=entry_price + 1.0,
            take_profit_2=entry_price + 2.0,
            stop_distance=1.0,
            risk_amount=risk_amount,
            position_size=risk_amount,
            reward_to_risk=reward_to_risk,
        )

    monkeypatch.setattr("trading_bot.backtest.get_trade_decision", fake_decision)
    monkeypatch.setattr("trading_bot.backtest.build_trade_plan", fake_trade_plan)

    baseline = run_ltf_backtest(
        df,
        max_daily_loss_fraction=1.0,
        cooldown_bars_after_loss=0,
        loss_streak_for_cooldown=1,
    )
    cooled = run_ltf_backtest(
        df,
        max_daily_loss_fraction=1.0,
        cooldown_bars_after_loss=3,
        loss_streak_for_cooldown=1,
    )

    assert baseline.trade_count > cooled.trade_count
    assert cooled.cooldown_events > 0
