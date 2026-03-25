import math

import pytest

from trading_bot.risk import build_trade_plan


def test_build_trade_plan_for_long_position() -> None:
    plan = build_trade_plan(
        side="long",
        entry_price=2400.0,
        atr_value=10.0,
        account_balance=1000.0,
        risk_fraction=0.01,
    )

    assert plan.side == "long"
    assert plan.risk_amount == 10.0
    assert plan.stop_distance == 15.0
    assert plan.stop_loss == 2385.0
    assert plan.take_profit_1 == 2415.0
    assert plan.take_profit_2 == 2430.0
    assert math.isclose(plan.position_size, 10.0 / 15.0)


def test_build_trade_plan_for_short_position() -> None:
    plan = build_trade_plan(
        side="short",
        entry_price=2400.0,
        atr_value=8.0,
        account_balance=2000.0,
        risk_fraction=0.02,
        atr_multiplier=2.0,
        reward_to_risk=3.0,
    )

    assert plan.side == "short"
    assert plan.risk_amount == 40.0
    assert plan.stop_distance == 16.0
    assert plan.stop_loss == 2416.0
    assert plan.take_profit_1 == 2384.0
    assert plan.take_profit_2 == 2352.0
    assert math.isclose(plan.position_size, 2.5)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("side", {"side": "buy"}),
        ("entry_price", {"entry_price": 0.0}),
        ("atr_value", {"atr_value": 0.0}),
        ("account_balance", {"account_balance": 0.0}),
        ("risk_fraction", {"risk_fraction": 0.0}),
        ("atr_multiplier", {"atr_multiplier": 0.0}),
        ("reward_to_risk", {"reward_to_risk": 0.0}),
    ],
)
def test_build_trade_plan_validates_inputs(field: str, kwargs: dict[str, float | str]) -> None:
    base_kwargs = {
        "side": "long",
        "entry_price": 2400.0,
        "atr_value": 10.0,
        "account_balance": 1000.0,
        "risk_fraction": 0.01,
        "atr_multiplier": 1.5,
        "reward_to_risk": 2.0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=field):
        build_trade_plan(**base_kwargs)
