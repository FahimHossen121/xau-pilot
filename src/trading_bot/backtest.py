"""Paper-only historical replay helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from trading_bot.risk import TradePlan, build_trade_plan
from trading_bot.strategies import (
    DEFAULT_ATR_FLOOR_RATIO,
    DEFAULT_LTF_THRESHOLD,
    DEFAULT_STRUCTURE_LOOKBACK,
    DEFAULT_LTF_WEIGHTS,
    add_ltf_features,
    get_trade_decision,
    get_htf_state_series,
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
    transaction_cost: float
    htf_state: str | None = None
    session: str | None = None
    strategy_mode: str | None = None


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
    total_r_multiple: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    max_drawdown_pct: float
    max_daily_loss_fraction: float
    cooldown_bars_after_loss: int
    loss_streak_for_cooldown: int
    cooldown_events: int
    daily_loss_lockout_days: int
    one_open_position_rule: bool
    used_htf_filter: bool
    htf_rule: str | None
    trades: list[BacktestTrade]


@dataclass
class _OpenPosition:
    trade_plan: TradePlan
    entry_time: pd.Timestamp
    risk_amount: float
    htf_state: str | None = None
    session: str | None = None
    strategy_mode: str | None = None


@dataclass(frozen=True)
class _TradeAggregate:
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    total_pnl: float
    total_r_multiple: float
    average_r_multiple: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float


@dataclass(frozen=True)
class _RiskRuntimeUpdate:
    balance: float
    day_realized_pnl: float
    day_locked: bool
    consecutive_losses: int
    cooldown_until_index: int
    cooldown_events: int


def _resolve_exit(
    position: _OpenPosition,
    candle: pd.Series,
    *,
    timestamp: pd.Timestamp,
    spread: float = 0.0,
    slippage: float = 0.0,
) -> BacktestTrade | None:
    plan = position.trade_plan
    high = float(candle["high"])
    low = float(candle["low"])
    per_side_cost = (spread / 2.0) + slippage

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

    transaction_cost = 2.0 * per_side_cost * plan.position_size
    pnl -= transaction_cost

    return BacktestTrade(
        entry_time=position.entry_time,
        exit_time=timestamp,
        side=plan.side,
        entry_price=plan.entry_price,
        exit_price=exit_price,
        pnl=pnl,
        r_multiple=pnl / position.risk_amount,
        exit_reason=reason,
        transaction_cost=transaction_cost,
        htf_state=position.htf_state,
        session=position.session,
        strategy_mode=position.strategy_mode,
    )


def _register_closed_trade(
    *,
    closed_trade: BacktestTrade,
    trades: list[BacktestTrade],
    balance: float,
    day_realized_pnl: float,
    day_start_balance: float,
    current_day_value: date,
    lockout_days: set[date],
    consecutive_losses: int,
    index: int,
    cooldown_until_index: int,
    cooldown_bars_after_loss: int,
    loss_streak_for_cooldown: int,
    cooldown_events: int,
    max_daily_loss_fraction: float,
) -> _RiskRuntimeUpdate:
    trades.append(closed_trade)
    balance += closed_trade.pnl
    day_realized_pnl += closed_trade.pnl

    if closed_trade.pnl < 0:
        consecutive_losses += 1
        if (
            cooldown_bars_after_loss > 0
            and consecutive_losses >= loss_streak_for_cooldown
        ):
            cooldown_until_index = max(
                cooldown_until_index,
                index + cooldown_bars_after_loss,
            )
            cooldown_events += 1
            consecutive_losses = 0
    else:
        consecutive_losses = 0

    day_locked = False
    daily_loss_limit = day_start_balance * max_daily_loss_fraction
    if day_realized_pnl <= -daily_loss_limit:
        day_locked = True
        lockout_days.add(current_day_value)

    return _RiskRuntimeUpdate(
        balance=balance,
        day_realized_pnl=day_realized_pnl,
        day_locked=day_locked,
        consecutive_losses=consecutive_losses,
        cooldown_until_index=cooldown_until_index,
        cooldown_events=cooldown_events,
    )


def _summarize_trades(trades: list[BacktestTrade]) -> _TradeAggregate:
    trade_count = len(trades)
    win_count = sum(1 for trade in trades if trade.pnl > 0)
    loss_count = sum(1 for trade in trades if trade.pnl < 0)
    total_pnl = sum(trade.pnl for trade in trades)
    total_r_multiple = sum(trade.r_multiple for trade in trades)
    average_r_multiple = (total_r_multiple / trade_count) if trade_count else 0.0
    win_rate = (win_count / trade_count) if trade_count else 0.0
    gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = sum(trade.pnl for trade in trades if trade.pnl < 0)
    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss < 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    average_win = (gross_profit / win_count) if win_count else 0.0
    average_loss = (gross_loss / loss_count) if loss_count else 0.0

    return _TradeAggregate(
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_r_multiple=total_r_multiple,
        average_r_multiple=average_r_multiple,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
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
    spread: float = 0.0,
    slippage: float = 0.0,
    max_daily_loss_fraction: float = 0.03,
    cooldown_bars_after_loss: int = 3,
    loss_streak_for_cooldown: int = 2,
    use_htf_filter: bool = False,
    htf_rule: str = "4H",
    htf_volatile_atr_ratio: float = 0.012,
    weights: dict[str, float] | None = None,
) -> BacktestResult:
    """Replay candles, open on the next bar, and exit on stop or TP2."""
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive.")
    if spread < 0:
        raise ValueError("spread must be non-negative.")
    if slippage < 0:
        raise ValueError("slippage must be non-negative.")
    if not 0 < max_daily_loss_fraction <= 1:
        raise ValueError("max_daily_loss_fraction must be between 0 and 1.")
    if cooldown_bars_after_loss < 0:
        raise ValueError("cooldown_bars_after_loss must be non-negative.")
    if loss_streak_for_cooldown < 1:
        raise ValueError("loss_streak_for_cooldown must be at least 1.")

    feature_frame = add_ltf_features(
        df,
        structure_lookback=structure_lookback,
        weights=weights or DEFAULT_LTF_WEIGHTS,
    )
    htf_states = (
        get_htf_state_series(
            df,
            rule=htf_rule,
            volatile_atr_ratio=htf_volatile_atr_ratio,
        )
        if use_htf_filter
        else None
    )

    balance = initial_balance
    trades: list[BacktestTrade] = []
    open_position: _OpenPosition | None = None
    current_day: date | None = None
    day_start_balance = initial_balance
    day_realized_pnl = 0.0
    day_locked = False
    lockout_days: set[date] = set()
    consecutive_losses = 0
    cooldown_until_index = -1
    cooldown_events = 0
    warmup_bars = max(200, structure_lookback)
    for index in range(warmup_bars + 1, len(feature_frame)):
        current_row = feature_frame.iloc[index]
        current_time = feature_frame.index[index]
        current_day_value = current_time.date()

        if current_day != current_day_value:
            current_day = current_day_value
            day_start_balance = balance
            day_realized_pnl = 0.0
            day_locked = False

        if open_position is not None:
            closed_trade = _resolve_exit(
                open_position,
                current_row,
                timestamp=current_time,
                spread=spread,
                slippage=slippage,
            )
            if closed_trade is not None:
                runtime = _register_closed_trade(
                    closed_trade=closed_trade,
                    trades=trades,
                    balance=balance,
                    day_realized_pnl=day_realized_pnl,
                    day_start_balance=day_start_balance,
                    current_day_value=current_day_value,
                    lockout_days=lockout_days,
                    consecutive_losses=consecutive_losses,
                    index=index,
                    cooldown_until_index=cooldown_until_index,
                    cooldown_bars_after_loss=cooldown_bars_after_loss,
                    loss_streak_for_cooldown=loss_streak_for_cooldown,
                    cooldown_events=cooldown_events,
                    max_daily_loss_fraction=max_daily_loss_fraction,
                )
                balance = runtime.balance
                day_realized_pnl = runtime.day_realized_pnl
                day_locked = runtime.day_locked
                consecutive_losses = runtime.consecutive_losses
                cooldown_until_index = runtime.cooldown_until_index
                cooldown_events = runtime.cooldown_events
                open_position = None

        if open_position is not None:
            continue
        if day_locked:
            continue
        if index <= cooldown_until_index:
            continue

        signal_slice = feature_frame.iloc[:index]
        htf_state = str(htf_states.iloc[index - 1]) if htf_states is not None else None
        decision = get_trade_decision(
            signal_slice,
            timestamp=feature_frame.index[index - 1],
            htf_state=htf_state,
            threshold=threshold,
            atr_floor_ratio=atr_floor_ratio,
            structure_lookback=structure_lookback,
            weights=weights,
        )
        if not decision.tradable:
            continue

        signal_row = feature_frame.iloc[index - 1]
        side = "long" if decision.bias == "bullish" else "short"

        trade_plan = build_trade_plan(
            side=side,
            entry_price=float(current_row["open"]),
            atr_value=float(signal_row["atr_14"]),
            account_balance=balance,
            risk_fraction=risk_fraction,
            atr_multiplier=decision.atr_multiplier,
            reward_to_risk=decision.reward_to_risk,
        )
        open_position = _OpenPosition(
            trade_plan=trade_plan,
            entry_time=current_time,
            risk_amount=trade_plan.risk_amount,
            htf_state=htf_state,
            session=decision.session.value,
            strategy_mode=decision.strategy_mode,
        )

        closed_trade = _resolve_exit(
            open_position,
            current_row,
            timestamp=current_time,
            spread=spread,
            slippage=slippage,
        )
        if closed_trade is not None:
            runtime = _register_closed_trade(
                closed_trade=closed_trade,
                trades=trades,
                balance=balance,
                day_realized_pnl=day_realized_pnl,
                day_start_balance=day_start_balance,
                current_day_value=current_day_value,
                lockout_days=lockout_days,
                consecutive_losses=consecutive_losses,
                index=index,
                cooldown_until_index=cooldown_until_index,
                cooldown_bars_after_loss=cooldown_bars_after_loss,
                loss_streak_for_cooldown=loss_streak_for_cooldown,
                cooldown_events=cooldown_events,
                max_daily_loss_fraction=max_daily_loss_fraction,
            )
            balance = runtime.balance
            day_realized_pnl = runtime.day_realized_pnl
            day_locked = runtime.day_locked
            consecutive_losses = runtime.consecutive_losses
            cooldown_until_index = runtime.cooldown_until_index
            cooldown_events = runtime.cooldown_events
            open_position = None

    if open_position is not None:
        last_row = feature_frame.iloc[-1]
        last_close = float(last_row["close"])
        pnl = (
            (last_close - open_position.trade_plan.entry_price) * open_position.trade_plan.position_size
            if open_position.trade_plan.side == "long"
            else (open_position.trade_plan.entry_price - last_close) * open_position.trade_plan.position_size
        )
        transaction_cost = 2.0 * ((spread / 2.0) + slippage) * open_position.trade_plan.position_size
        pnl -= transaction_cost
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
                transaction_cost=transaction_cost,
                htf_state=open_position.htf_state,
                session=open_position.session,
                strategy_mode=open_position.strategy_mode,
            )
        )
        balance += pnl

    aggregate = _summarize_trades(trades)
    total_pnl = balance - initial_balance

    running_balance = initial_balance
    peak_balance = initial_balance
    max_drawdown_pct = 0.0
    for trade in trades:
        running_balance += trade.pnl
        peak_balance = max(peak_balance, running_balance)
        drawdown_pct = (
            (peak_balance - running_balance) / peak_balance if peak_balance else 0.0
        )
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

    return BacktestResult(
        initial_balance=initial_balance,
        final_balance=balance,
        total_pnl=total_pnl,
        total_return_pct=(total_pnl / initial_balance) if initial_balance else 0.0,
        trade_count=aggregate.trade_count,
        win_count=aggregate.win_count,
        loss_count=aggregate.loss_count,
        win_rate=aggregate.win_rate,
        average_r_multiple=aggregate.average_r_multiple,
        total_r_multiple=aggregate.total_r_multiple,
        gross_profit=aggregate.gross_profit,
        gross_loss=aggregate.gross_loss,
        profit_factor=aggregate.profit_factor,
        average_win=aggregate.average_win,
        average_loss=aggregate.average_loss,
        max_drawdown_pct=max_drawdown_pct,
        max_daily_loss_fraction=max_daily_loss_fraction,
        cooldown_bars_after_loss=cooldown_bars_after_loss,
        loss_streak_for_cooldown=loss_streak_for_cooldown,
        cooldown_events=cooldown_events,
        daily_loss_lockout_days=len(lockout_days),
        one_open_position_rule=True,
        used_htf_filter=use_htf_filter,
        htf_rule=htf_rule if use_htf_filter else None,
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
    spread: float = 0.0,
    slippage: float = 0.0,
    max_daily_loss_fraction: float = 0.03,
    cooldown_bars_after_loss: int = 3,
    loss_streak_for_cooldown: int = 2,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_incomplete_last_candle: bool = True,
    use_htf_filter: bool = False,
    htf_rule: str = "4H",
    htf_volatile_atr_ratio: float = 0.012,
    weights: dict[str, float] | None = None,
) -> BacktestResult:
    """Fetch MT5 candles and run the paper-only LTF replay."""
    from trading_bot.data import get_candles, get_candles_range

    candles = (
        get_candles_range(symbol, timeframe, start_time, end_time)
        if start_time is not None and end_time is not None
        else get_candles(symbol, timeframe, count)
    )
    if drop_incomplete_last_candle and len(candles) > 1:
        candles = candles.iloc[:-1]

    return run_ltf_backtest(
        candles,
        initial_balance=initial_balance,
        risk_fraction=risk_fraction,
        threshold=threshold,
        atr_floor_ratio=atr_floor_ratio,
        structure_lookback=structure_lookback,
        atr_multiplier=atr_multiplier,
        reward_to_risk=reward_to_risk,
        spread=spread,
        slippage=slippage,
        max_daily_loss_fraction=max_daily_loss_fraction,
        cooldown_bars_after_loss=cooldown_bars_after_loss,
        loss_streak_for_cooldown=loss_streak_for_cooldown,
        use_htf_filter=use_htf_filter,
        htf_rule=htf_rule,
        htf_volatile_atr_ratio=htf_volatile_atr_ratio,
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
        f"Total R multiple: {result.total_r_multiple:.2f}",
        f"Gross profit: {result.gross_profit:.2f}",
        f"Gross loss: {result.gross_loss:.2f}",
        f"Profit factor: {result.profit_factor:.2f}",
        f"Average win: {result.average_win:.2f}",
        f"Average loss: {result.average_loss:.2f}",
        f"Max drawdown: {result.max_drawdown_pct * 100:.2f}%",
        f"Max daily loss limit: {result.max_daily_loss_fraction * 100:.2f}%",
        f"Cooldown bars after loss streak: {result.cooldown_bars_after_loss}",
        f"Loss streak for cooldown: {result.loss_streak_for_cooldown}",
        f"Cooldown events: {result.cooldown_events}",
        f"Daily lockout days: {result.daily_loss_lockout_days}",
        f"One open position rule: {result.one_open_position_rule}",
        f"HTF filter enabled: {result.used_htf_filter}",
        f"HTF rule: {result.htf_rule or 'none'}",
    ]
    return "\n".join(lines)


def backtest_result_to_row(
    result: BacktestResult,
    *,
    scenario_name: str,
    symbol: str,
    timeframe: str,
    candle_count: int,
    window_start: str | None = None,
    window_end: str | None = None,
    risk_fraction: float,
    spread: float,
    slippage: float,
) -> dict[str, float | int | str | bool | None]:
    """Flatten a backtest result into a single summary row for CSV export."""
    return {
        "scenario_name": scenario_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_count": candle_count,
        "window_start": window_start,
        "window_end": window_end,
        "risk_fraction": risk_fraction,
        "spread": spread,
        "slippage": slippage,
        "max_daily_loss_fraction": result.max_daily_loss_fraction,
        "cooldown_bars_after_loss": result.cooldown_bars_after_loss,
        "loss_streak_for_cooldown": result.loss_streak_for_cooldown,
        "cooldown_events": result.cooldown_events,
        "daily_loss_lockout_days": result.daily_loss_lockout_days,
        "one_open_position_rule": result.one_open_position_rule,
        "used_htf_filter": result.used_htf_filter,
        "htf_rule": result.htf_rule,
        "initial_balance": result.initial_balance,
        "final_balance": result.final_balance,
        "total_pnl": result.total_pnl,
        "total_return_pct": result.total_return_pct,
        "trade_count": result.trade_count,
        "win_count": result.win_count,
        "loss_count": result.loss_count,
        "win_rate": result.win_rate,
        "average_r_multiple": result.average_r_multiple,
        "total_r_multiple": result.total_r_multiple,
        "gross_profit": result.gross_profit,
        "gross_loss": result.gross_loss,
        "profit_factor": result.profit_factor,
        "average_win": result.average_win,
        "average_loss": result.average_loss,
        "max_drawdown_pct": result.max_drawdown_pct,
    }


def backtest_trades_to_frame(
    result: BacktestResult,
    *,
    scenario_name: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Convert the executed trades into a CSV-friendly dataframe."""
    columns = [
        "scenario_name",
        "symbol",
        "timeframe",
        "entry_time",
        "exit_time",
        "side",
        "session",
        "strategy_mode",
        "entry_price",
        "exit_price",
        "pnl",
        "r_multiple",
        "exit_reason",
        "transaction_cost",
        "htf_state",
    ]
    rows: list[dict[str, float | str | None]] = []
    for trade in result.trades:
        rows.append(
            {
                "scenario_name": scenario_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "side": trade.side,
                "session": trade.session,
                "strategy_mode": trade.strategy_mode,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "r_multiple": trade.r_multiple,
                "exit_reason": trade.exit_reason,
                "transaction_cost": trade.transaction_cost,
                "htf_state": trade.htf_state,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def backtest_session_stats_to_frame(
    result: BacktestResult,
    *,
    scenario_name: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Aggregate executed trades by session and strategy mode for CSV export."""
    rows: list[dict[str, float | int | str | bool | None]] = []
    trades_frame = backtest_trades_to_frame(
        result,
        scenario_name=scenario_name,
        symbol=symbol,
        timeframe=timeframe,
    )
    if trades_frame.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "symbol",
                "timeframe",
                "session",
                "strategy_mode",
                "trade_count",
                "win_count",
                "loss_count",
                "win_rate",
                "total_pnl",
                "total_r_multiple",
                "average_r_multiple",
                "profit_factor",
                "average_win",
                "average_loss",
            ]
        )

    for (session, strategy_mode), group in trades_frame.groupby(
        ["session", "strategy_mode"],
        dropna=False,
    ):
        trade_subset = [
            BacktestTrade(
                entry_time=pd.Timestamp(row.entry_time),
                exit_time=pd.Timestamp(row.exit_time),
                side=str(row.side),
                entry_price=float(row.entry_price),
                exit_price=float(row.exit_price),
                pnl=float(row.pnl),
                r_multiple=float(row.r_multiple),
                exit_reason=str(row.exit_reason),
                transaction_cost=float(row.transaction_cost),
                htf_state=None if pd.isna(row.htf_state) else str(row.htf_state),
                session=None if pd.isna(row.session) else str(row.session),
                strategy_mode=None if pd.isna(row.strategy_mode) else str(row.strategy_mode),
            )
            for row in group.itertuples(index=False)
        ]
        aggregate = _summarize_trades(trade_subset)
        rows.append(
            {
                "scenario_name": scenario_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "session": session,
                "strategy_mode": strategy_mode,
                "trade_count": aggregate.trade_count,
                "win_count": aggregate.win_count,
                "loss_count": aggregate.loss_count,
                "win_rate": aggregate.win_rate,
                "total_pnl": aggregate.total_pnl,
                "total_r_multiple": aggregate.total_r_multiple,
                "average_r_multiple": aggregate.average_r_multiple,
                "profit_factor": aggregate.profit_factor,
                "average_win": aggregate.average_win,
                "average_loss": aggregate.average_loss,
            }
        )

    return pd.DataFrame(rows)
