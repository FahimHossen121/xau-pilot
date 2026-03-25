from trading_bot.config import Settings


def main() -> None:
    settings = Settings.from_env()
    print(f"Template booted in {settings.app_mode} mode for symbol {settings.symbol}.")


if __name__ == "__main__":
    main()
