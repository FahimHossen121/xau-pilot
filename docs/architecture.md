# Architecture Template (Paper-First)

## Runtime Modes

- paper: default mode, demo account + simulation guards
- live: disabled until validation milestones are met

## Core Modules

- config: environment parsing and validation
- data: market and macro pipelines
- strategies: HTF and LTF signal layers
- risk: sizing, stops, daily protections
- adapters: MT5 and model providers (Gemini)
- execution: order routing and state transitions
- reporting: trade journal, metrics, alerts

## Safety Controls

- hard daily loss limit
- max risk per trade
- spread/slippage filter
- high-impact event blackout window
- kill switch after consecutive losses

## Observability

- structured logs
- heartbeat and connection checks
- daily PnL summary
- failure notifications
