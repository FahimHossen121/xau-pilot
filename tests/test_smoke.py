from pathlib import Path

import pytest

from trading_bot.config import Settings


ENV_KEYS = [
    "APP_MODE",
    "LOG_LEVEL",
    "TIMEZONE",
    "SYMBOL",
    "MAX_RISK_PER_TRADE",
    "MAX_DAILY_LOSS",
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "ENABLE_TRADING",
]


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)

    settings = Settings.from_env(dotenv_path=Path("tests/does-not-exist.env"))

    assert settings.app_mode == "paper"
    assert settings.symbol == "XAUUSD"
    assert settings.max_risk_per_trade == 0.01
    assert settings.enable_trading is False


def test_settings_loads_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    env_path = Path("tests/.env.test")
    env_path.write_text(
        "\n".join(
            [
                "APP_MODE=paper",
                "LOG_LEVEL=debug",
                "TIMEZONE=Asia/Dhaka",
                "SYMBOL=xauusd",
                "MAX_RISK_PER_TRADE=0.02",
                "MAX_DAILY_LOSS=0.05",
                "MT5_LOGIN=123456",
                "MT5_PASSWORD=test-password",
                "MT5_SERVER=HFMarketsGlobal-Demo",
                "ENABLE_TRADING=true",
            ]
        ),
        encoding="utf-8",
    )

    try:
        settings = Settings.from_env(dotenv_path=env_path)
    finally:
        env_path.unlink(missing_ok=True)

    assert settings.log_level == "DEBUG"
    assert settings.timezone == "Asia/Dhaka"
    assert settings.symbol == "XAUUSD"
    assert settings.max_risk_per_trade == 0.02
    assert settings.max_daily_loss == 0.05
    assert settings.mt5_login == 123456
    assert settings.mt5_server == "HFMarketsGlobal-Demo"
    assert settings.enable_trading is True


def test_settings_rejects_partial_mt5_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    env_path = Path("tests/.env.partial.test")
    env_path.write_text("MT5_LOGIN=123456\n", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="must be set together"):
            Settings.from_env(dotenv_path=env_path)
    finally:
        env_path.unlink(missing_ok=True)
