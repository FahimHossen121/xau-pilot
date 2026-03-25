from trading_bot.config import Settings


def test_settings_defaults() -> None:
    settings = Settings.from_env()
    assert settings.app_mode in {"paper", "live"}
    assert settings.symbol
