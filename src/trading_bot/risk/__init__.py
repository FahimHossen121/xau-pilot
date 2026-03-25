"""Risk sizing and trade-level planning helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradePlan:
    side: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    stop_distance: float
    risk_amount: float
    position_size: float
    reward_to_risk: float


def build_trade_plan(
    *,
    side: str,
    entry_price: float,
    atr_value: float,
    account_balance: float,
    risk_fraction: float,
    atr_multiplier: float = 1.5,
    reward_to_risk: float = 2.0,
) -> TradePlan:
    """Build a paper-trading risk plan from ATR and account-level risk settings."""
    normalized_side = side.lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'.")

    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")

    if atr_value <= 0:
        raise ValueError("atr_value must be positive.")

    if account_balance <= 0:
        raise ValueError("account_balance must be positive.")

    if not 0 < risk_fraction <= 1:
        raise ValueError("risk_fraction must be between 0 and 1.")

    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive.")

    if reward_to_risk <= 0:
        raise ValueError("reward_to_risk must be positive.")

    stop_distance = atr_value * atr_multiplier
    risk_amount = account_balance * risk_fraction
    position_size = risk_amount / stop_distance

    if normalized_side == "long":
        stop_loss = entry_price - stop_distance
        take_profit_1 = entry_price + stop_distance
        take_profit_2 = entry_price + (stop_distance * reward_to_risk)
    else:
        stop_loss = entry_price + stop_distance
        take_profit_1 = entry_price - stop_distance
        take_profit_2 = entry_price - (stop_distance * reward_to_risk)

    return TradePlan(
        side=normalized_side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        stop_distance=stop_distance,
        risk_amount=risk_amount,
        position_size=position_size,
        reward_to_risk=reward_to_risk,
    )
