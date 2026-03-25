from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_mode: str
    symbol: str
    enable_trading: bool

    @classmethod
    def from_env(cls) -> "Settings":
        app_mode = os.getenv("APP_MODE", "paper").strip().lower()
        if app_mode not in {"paper", "live"}:
            raise ValueError("APP_MODE must be 'paper' or 'live'.")

        symbol = os.getenv("SYMBOL", "XAUUSD").strip().upper()
        enable_trading = os.getenv("ENABLE_TRADING", "false").strip().lower() == "true"
        return cls(app_mode=app_mode, symbol=symbol, enable_trading=enable_trading)
