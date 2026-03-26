from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def _load_local_dotenv(dotenv_path: Path | None = None) -> None:
    resolved_path = dotenv_path or (Path.cwd() / ".env")
    if resolved_path.exists():
        load_dotenv(dotenv_path=resolved_path, override=False)


def _get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = _get_str(name, str(default).lower()).lower()
    return raw in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = _get_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number.") from exc


def _get_optional_int(name: str) -> int | None:
    raw = _get_str(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer.") from exc


def _get_csv_list(name: str) -> tuple[str, ...]:
    raw = _get_str(name)
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_mode: str
    log_level: str
    timezone: str
    symbol: str
    max_risk_per_trade: float
    max_daily_loss: float
    max_weekly_loss: float
    max_account_drawdown: float
    min_balance_fraction: float
    mt5_login: int | None
    mt5_password: str | None
    mt5_server: str | None
    news_provider: str
    rss_feed_urls: tuple[str, ...]
    brave_api_key: str | None
    gemini_api_key: str | None
    gemini_model: str
    ai_htf_refresh_hours: int
    brave_news_freshness: str
    brave_news_results_per_query: int
    enable_trading: bool

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "Settings":
        _load_local_dotenv(dotenv_path)

        app_mode = _get_str("APP_MODE", "paper").lower()
        if app_mode not in {"paper", "live"}:
            raise ValueError("APP_MODE must be 'paper' or 'live'.")

        settings = cls(
            app_mode=app_mode,
            log_level=_get_str("LOG_LEVEL", "INFO").upper(),
            timezone=_get_str("TIMEZONE", "UTC"),
            symbol=_get_str("SYMBOL", "XAUUSD").upper(),
            max_risk_per_trade=_get_float("MAX_RISK_PER_TRADE", 0.01),
            max_daily_loss=_get_float("MAX_DAILY_LOSS", 0.03),
            max_weekly_loss=_get_float("MAX_WEEKLY_LOSS", 0.05),
            max_account_drawdown=_get_float("MAX_ACCOUNT_DRAWDOWN", 0.12),
            min_balance_fraction=_get_float("MIN_BALANCE_FRACTION", 0.70),
            mt5_login=_get_optional_int("MT5_LOGIN"),
            mt5_password=_get_str("MT5_PASSWORD") or None,
            mt5_server=_get_str("MT5_SERVER") or None,
            news_provider=_get_str("NEWS_PROVIDER", "none").lower(),
            rss_feed_urls=_get_csv_list("RSS_FEED_URLS"),
            brave_api_key=_get_str("BRAVE_API_KEY") or None,
            gemini_api_key=_get_str("GEMINI_API_KEY") or None,
            gemini_model=_get_str("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            ai_htf_refresh_hours=int(_get_float("AI_HTF_REFRESH_HOURS", 1.0)),
            brave_news_freshness=_get_str("BRAVE_NEWS_FRESHNESS", "pd").lower(),
            brave_news_results_per_query=int(_get_float("BRAVE_NEWS_RESULTS_PER_QUERY", 5.0)),
            enable_trading=_get_bool("ENABLE_TRADING", False),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if not self.symbol:
            raise ValueError("SYMBOL cannot be empty.")

        if not 0 < self.max_risk_per_trade <= 1:
            raise ValueError("MAX_RISK_PER_TRADE must be between 0 and 1.")

        if not 0 < self.max_daily_loss <= 1:
            raise ValueError("MAX_DAILY_LOSS must be between 0 and 1.")

        if not 0 < self.max_weekly_loss <= 1:
            raise ValueError("MAX_WEEKLY_LOSS must be between 0 and 1.")

        if not 0 < self.max_account_drawdown <= 1:
            raise ValueError("MAX_ACCOUNT_DRAWDOWN must be between 0 and 1.")

        if not 0 < self.min_balance_fraction <= 1:
            raise ValueError("MIN_BALANCE_FRACTION must be between 0 and 1.")

        if self.ai_htf_refresh_hours < 1:
            raise ValueError("AI_HTF_REFRESH_HOURS must be at least 1.")

        if self.brave_news_freshness not in {"pd", "pw", "pm", "py"}:
            raise ValueError("BRAVE_NEWS_FRESHNESS must be one of: pd, pw, pm, py.")

        if self.brave_news_results_per_query < 1:
            raise ValueError("BRAVE_NEWS_RESULTS_PER_QUERY must be at least 1.")

        mt5_values = [self.mt5_login, self.mt5_password, self.mt5_server]
        if any(value is not None for value in mt5_values) and not all(mt5_values):
            raise ValueError(
                "MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER must be set together."
            )

        if self.news_provider not in {"none", "rss", "brave"}:
            raise ValueError("NEWS_PROVIDER must be one of: none, rss, brave.")
