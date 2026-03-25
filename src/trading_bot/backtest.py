"""Paper-only historical replay helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.risk import TradePlan, build_trade_plan
from trading_bot.strategies import (
    DEFAULT_ATR_FLOOR_RATIO,
    DEFAULT_LTF_THRESHOLD,
    DEFAULT_STRUCTURE_LOOKBACK,
    DEFAULT_LTF_WEIGHTS,
    add_ltf_features,
)


@dataclass(frozen=True)
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    r_multiple: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_return_pct: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    average_r_multiple: float
    trades: list[BacktestTrade]


@dataclass
class _OpenPosition:
    trade_plan: TradePlan
    entry_time: pd.Timestamp
    risk_amount: float


def _resolve_exit(
    position: _OpenPosition,
    candle: pd.Series,
    *,
    timestamp: pd.Timestamp,
) -> BacktestTrade | None:
    plan = position.trade_plan
    high = float(candle["high"])
    low = float(candle["low"])

    if plan.side == "long":
        hit_stop = low <= plan.stop_loss
        hit_target = high >= plan.take_profit_2

        if hit_stop and hit_target:
            exit_price = plan.stop_loss
            pnl = -position.risk_amount
            reason = "stop_loss_ambiguous"
        elif hit_stop:
            exit_price = plan.stop_loss
            pnl = -position.risk_amount
            reason = "stop_loss"
        elif hit_target:
            exit_price = plan.take_profit_2
            pnl = position.risk_amount * plan.reward_to_risk
            reason = "take_profit"
        else:
            return None
    else:
        hit_stop = high >= plan.stop_loss
        hit_target = low <= plan.take_profit_2

        if hit_stop and hit_target:
            exit_price = plan.stop_loss
            pnl = -position.risk_amount
            reason = "stop_loss_ambiguous"
        elif hit_stop:
            exit_price = plan.stop_loss
            pnl = -position.risk_amount
            reason = "stop_loss"
        elif hit_target:
            exit_price = plan.take_profit_2
            pnl = position.risk_amount * plan.reward_to_risk
            reason = "take_profit"
        else:
            return None

    return BacktestTrade(
        entry_time=position.entry_time,
        exit_time=timestamp,
        side=plan.side,
        entry_price=plan.entry_price,
        exit_price=exit_price,
        pnl=pnl,
        r_multiple=pnl / position.risk_amount,
        exit_reason=reason,
    )


def run_ltf_backtest(
    df: pd.DataFrame,
    *,
    initial_balance: float = 1000.0,
    risk_fraction: float = 0.01,
    threshold: float = DEFAULT_LTF_THRESHOLD,
    atr_floor_ratio: float = DEFAULT_ATR_FLOOR_RATIO,
    structure_lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
    atr_multiplier: float = 1.5,
    reward_to_risk: float = 2.0,
    weights: dict[str, float] | None = None,
) -> BacktestResult:
    """Replay candles, open on the next bar, and exit on stop or TP2."""
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive.")

    feature_frame = add_ltf_features(
        df,
        structure_lookback=structure_lookback,
        weights=weights or DEFAULT_LTF_WEIGHTS,
    )

    balance = initial_balance
    trades: list[BacktestTrade] = []
    open_position: _OpenPosition | None = None
    warmup_bars = max(200, structure_lookback)

    for index in range(warmup_bars + 1, len(feature_frame)):
        current_row = feature_frame.iloc[index]
        current_time = feature_frame.index[index]

        if open_position is not None:
            closed_trade = _resolve_exit(open_position, current_row, timestamp=current_time)
            if closed_trade is not None:
                trades.append(closed_trade)
                balance += closed_trade.pnl
                open_position = None

        if open_position is not None:
            continue

        signal_row = feature_frame.iloc[index - 1]
        atr_ratio = float(signal_row["atr_ratio"])
        score = float(signal_row["ltf_score"])

        if atr_ratio < atr_floor_ratio:
            continue

        if score >= threshold:
            side = "long"
        elif score <= -threshold:
            side = "short"
        else:
            continue

        trade_plan = build_trade_plan(
            side=side,
            entry_price=float(current_row["open"]),
            atr_value=float(signal_row["atr_14"]),
            account_balance=balance,
            risk_fraction=risk_fraction,
            atr_multiplier=atr_multiplier,
            reward_to_risk=reward_to_risk,
        )
        open_position = _OpenPosition(
            trade_plan=trade_plan,
            entry_time=current_time,
            risk_amount=trade_plan.risk_amount,
        )

        closed_trade = _resolve_exit(open_position, current_row, timestamp=current_time)
        if closed_trade is not None:
            trades.append(closed_trade)
            balance += closed_trade.pnl
            open_position = None

    if open_position is not None:
        last_row = feature_frame.iloc[-1]
        last_close = float(last_row["close"])
        pnl = (
            (last_close - open_position.trade_plan.entry_price) * open_position.trade_plan.position_size
            if open_position.trade_plan.side == "long"
            else (open_position.trade_plan.entry_price - last_close) * open_position.trade_plan.position_size
        )
        trades.append(
            BacktestTrade(
                entry_time=open_position.entry_time,
                exit_time=feature_frame.index[-1],
                side=open_position.trade_plan.side,
                entry_price=open_position.trade_plan.entry_price,
                exit_price=last_close,
                pnl=pnl,
                r_multiple=pnl / open_position.risk_amount,
                exit_reason="end_of_data",
            )
        )
        balance += pnl

    trade_count = len(trades)
    win_count = sum(1 for trade in trades if trade.pnl > 0)
    loss_count = sum(1 for trade in trades if trade.pnl < 0)
    total_pnl = balance - initial_balance
    average_r_multiple = (
        sum(trade.r_multiple for trade in trades) / trade_count if trade_count else 0.0
    )
    win_rate = (win_count / trade_count) if trade_count else 0.0

    return BacktestResult(
        initial_balance=initial_balance,
        final_balance=balance,
        total_pnl=total_pnl,
        total_return_pct=(total_pnl / initial_balance) if initial_balance else 0.0,
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        average_r_multiple=average_r_multiple,
        trades=trades,
    )


def run_mt5_ltf_backtest(
    *,
    symbol: str,
    timeframe: int,
    count: int,
    initial_balance: float = 1000.0,
    risk_fraction: float = 0.01,
    threshold: float = DEFAULT_LTF_THRESHOLD,
    atr_floor_ratio: float = DEFAULT_ATR_FLOOR_RATIO,
    structure_lookback: int = DEFAULT_STRUCTURE_LOOKBACK,
    atr_multiplier: float = 1.5,
    reward_to_risk: float = 2.0,
    weights: dict[str, float] | None = None,
) -> BacktestResult:
    """Fetch MT5 candles and run the paper-only LTF replay."""
    from trading_bot.data import get_candles

    candles = get_candles(symbol, timeframe, count)
    return run_ltf_backtest(
        candles,
        initial_balance=initial_balance,
        risk_fraction=risk_fraction,
        threshold=threshold,
        atr_floor_ratio=atr_floor_ratio,
        structure_lookback=structure_lookback,
        atr_multiplier=atr_multiplier,
        reward_to_risk=reward_to_risk,
        weights=weights,
    )


def format_backtest_summary(result: BacktestResult) -> str:
    """Return a short human-readable summary for terminal output."""
    lines = [
        "Backtest Summary",
        f"Initial balance: {result.initial_balance:.2f}",
        f"Final balance: {result.final_balance:.2f}",
        f"Total PnL: {result.total_pnl:.2f}",
        f"Return: {result.total_return_pct * 100:.2f}%",
        f"Trades: {result.trade_count}",
        f"Wins: {result.win_count}",
        f"Losses: {result.loss_count}",
        f"Win rate: {result.win_rate * 100:.2f}%",
        f"Average R multiple: {result.average_r_multiple:.2f}",
    ]
    return "\n".join(lines)
