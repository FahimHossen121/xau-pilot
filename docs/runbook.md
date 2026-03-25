# Runbook Template

## Local Setup

1. Install Python 3.11
2. Create and activate virtual environment
3. Install dependencies
4. Create `.env` from `.env.example`
5. Keep `APP_MODE=paper` and `ENABLE_TRADING=false`

## Startup Checklist

- MT5 desktop terminal open and logged into demo server
- XAUUSD visible in Market Watch
- Environment variables loaded
- Network connectivity healthy

## Incident Checklist

- MT5 disconnect: pause new orders, reconnect, verify account state
- API timeout: retry with backoff, then disable new entries
- data gap: halt strategy cycle until candles are continuous

## Promotion Gates (Paper -> Live)

- minimum forward-test duration met
- risk limits validated
- broker execution behavior confirmed
- manual approval documented
