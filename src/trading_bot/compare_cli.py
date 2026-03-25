from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_bot.backtest import (
    backtest_result_to_row,
    backtest_trades_to_frame,
    format_backtest_summary,
    run_mt5_ltf_backtest,
)
from trading_bot.config import Settings


@dataclass(frozen=True)
class Scenario:
    name: str
    timeframe: str
    count: int
    use_htf_filter: bool
    htf_rule: str | None


DEFAULT_SCENARIOS = [
    Scenario(name="m15_ltf_only", timeframe="M15", count=2000, use_htf_filter=False, htf_rule=None),
    Scenario(name="m15_htf_4h", timeframe="M15", count=2000, use_htf_filter=True, htf_rule="4H"),
    Scenario(name="m5_ltf_only", timeframe="M5", count=3000, use_htf_filter=False, htf_rule=None),
    Scenario(name="m5_htf_1h", timeframe="M5", count=3000, use_htf_filter=True, htf_rule="1H"),
]

TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
}


def _to_mt5_timeframe(label: str) -> int:
    import MetaTrader5 as mt5

    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }
    return mapping[label]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple MT5 backtest scenarios and export CSV reports.")
    parser.add_argument("--symbol", default=None, help="Trading symbol, default comes from .env")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance for each scenario")
    parser.add_argument("--risk", type=float, default=None, help="Risk fraction override")
    parser.add_argument("--spread", type=float, default=0.30, help="Absolute spread assumption")
    parser.add_argument("--slippage", type=float, default=0.05, help="Absolute slippage assumption per side")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where summary and trades CSV files will be written",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    symbol = args.symbol or settings.symbol
    risk_fraction = args.risk if args.risk is not None else settings.max_risk_per_trade

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: list[dict[str, float | int | str | bool | None]] = []
    trade_frames: list[pd.DataFrame] = []

    for scenario in DEFAULT_SCENARIOS:
        result = run_mt5_ltf_backtest(
            symbol=symbol,
            timeframe=_to_mt5_timeframe(scenario.timeframe),
            count=scenario.count,
            initial_balance=args.balance,
            risk_fraction=risk_fraction,
            spread=args.spread,
            slippage=args.slippage,
            use_htf_filter=scenario.use_htf_filter,
            htf_rule=scenario.htf_rule or "4H",
        )
        summary_rows.append(
            backtest_result_to_row(
                result,
                scenario_name=scenario.name,
                symbol=symbol,
                timeframe=scenario.timeframe,
                candle_count=scenario.count,
                risk_fraction=risk_fraction,
                spread=args.spread,
                slippage=args.slippage,
            )
        )
        trade_frames.append(
            backtest_trades_to_frame(
                result,
                scenario_name=scenario.name,
                symbol=symbol,
                timeframe=scenario.timeframe,
            )
        )
        print()
        print(f"[{scenario.name}]")
        print(format_backtest_summary(result))

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    summary_path = output_dir / f"backtest_summary_{timestamp}.csv"
    trades_path = output_dir / f"backtest_trades_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print()
    print(f"Summary CSV: {summary_path}")
    print(f"Trades CSV: {trades_path}")


if __name__ == "__main__":
    main()
