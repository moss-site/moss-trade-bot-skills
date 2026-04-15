<div align="center">
  <img src="images/banner.png" alt="Moss Trade Bot Factory Banner" width="100%">
</div>

# Moss Trade Bot Factory

> 🎉 **Moss Trade Bot Factory** is an intelligent cryptocurrency quantitative trading bot factory and strategy tuner. By simply describing your trading style in natural language, the system automatically creates a crypto trading bot, runs local backtests, and supports periodic reflective evolution.

🚀 [Features](#features) | ⚡ [Installation & CLI](#installation--cli) | 🧠 [Evolution Mechanism](#evolution-mechanism) | 📦 [Platform Integration](#platform-integration) | 🤝 [Contributing](#contributing)

## Overview

Moss Trade Bot Factory transforms natural language descriptions into fully functional cryptocurrency trading strategies. It bridges the gap between trading ideas and quantitative execution by automatically inferring parameters, running cross-margin backtests, and iteratively evolving the strategy based on performance reflection.

> **Disclaimer**: This framework is designed for research and educational purposes. Trading performance may vary based on market conditions, data quality, and non-deterministic factors. It is not intended as financial, investment, or trading advice.

<div align="center">
  <img src="images/features_overview.png" alt="Features Overview" width="90%">
</div>

## Features

### 🗣️ Natural Language to Strategy
Describe your trading style (e.g., "Trend following, conservative leverage, breakout strategy"), and the AI automatically infers direction, leverage, risk parameters, and technical indicators.

### 📈 Full Backtesting Engine
A robust local backtesting engine featuring cross-margin simulation, regime detection, and rolling positions. It provides comprehensive metrics including Sharpe ratio, max drawdown, and win rate.

### 🧬 Weekly Evolution Loop
The core innovation of this factory. The AI reflects on segmented backtest results, analyzes winning and losing trades, and micro-adjusts tactical parameters while keeping the core personality locked.

### 🛡️ Safety Guardrails
Built-in safety mechanisms including leverage limits (max 150x), mandatory wide stop-losses for high leverage, and confirmation gates for live trading.

## System Architecture

The framework decomposes the complex process of strategy creation into a streamlined pipeline:

<div align="center">
  <img src="images/architecture.png" alt="System Architecture" width="100%">
</div>

### Signal Decision System

The core decision engine evaluates multiple market dimensions and normalizes them into a composite signal score:

<div align="center">
  <img src="images/signal_system.png" alt="Signal Decision System" width="90%">
</div>

- **Trend**: EMA crossover and Supertrend direction.
- **Momentum**: RSI and MACD oscillators.
- **Mean Reversion**: Bollinger Bands regression.
- **Volume**: OBV and volume-price correlation.
- **Volatility**: ATR breakout and contraction.

## Evolution Mechanism

Evolution is not a separate step after backtesting, but an embedded process during the backtest:

<div align="center">
  <img src="images/evolution_mechanism.png" alt="Evolution Mechanism" width="90%">
</div>

The AI applies **7 Reflection Principles** to analyze each segment's performance:
1. Look at the big picture before details.
2. Analyze why winning trades succeeded.
3. Analyze why losing trades failed.
4. Identify specific parameter issues.
5. Micro-adjust rather than reset (Tactical drift bounded to ±30%).
6. Maintain momentum from previous adjustments.
7. Ensure continuous adaptation (cannot remain unchanged for >3 rounds).

*Note: Personality parameters (bias, leverage, risk) are strictly locked during evolution.*

## Installation & CLI

### Prerequisites
- Python 3.x
- `pandas>=2.0.0`, `numpy>=1.24.0`, `ccxt>=4.0.0`, `scipy>=1.11.0`

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/moss-site/moss-trade-bot-skills.git
cd moss-trade-bot-skills/moss-trade-bot-factory-1.0.22-1.0.21/scripts
pip install -r requirements.txt
```

### Data Preparation

Fetch historical data for backtesting (Default: Binance USDT-M):

```bash
python3 fetch_data.py --symbol BTC/USDT --timeframe 15m
```

### Running Backtests

**Standard Backtest:**
```bash
python3 run_backtest.py --data <CSV_PATH> --params-file /tmp/bot_params.json --capital 10000 --output /tmp/backtest_result.json
```

**Evolution Backtest (Recommended):**
```bash
python3 run_evolve_backtest.py \
  --data <CSV_PATH> \
  --params-file /tmp/bot_params.json \
  --segment-bars 672 \
  --capital 10000 \
  --output /tmp/evolve_baseline.json
```

## Platform Integration (Optional)

The factory supports optional integration with the Moss platform for verification and simulated live trading. **All operations are local-first by default.**

### 1. Pair Code Binding
Bind your local environment to the platform using a Pair Code:
```bash
python3 live_trade.py bind \
  --platform-url "https://ai.moss.site" \
  --pair-code "<pair_code>" \
  --name "<Bot Name>" --persona "<Style>" --description "<Description>" \
  --save ~/.moss-trade-bot/agent_creds.json
```

### 2. Upload Verification
Upload your evolution backtest results for server-side validation:
```bash
python3 package_upload.py \
  --bot-name-zh "<中文名称>" --bot-name-en "<English Name>" \
  --bot-personality-zh "<中文风格>" --bot-personality-en "<English Persona>" \
  --bot-description-zh "<中文描述>" --bot-description-en "<English Description>" \
  --params-file /tmp/bot_params.json \
  --fingerprint-file /tmp/fingerprint.json \
  --result-file /tmp/evolve_result_final.json \
  --platform-url https://ai.moss.site \
  --creds ~/.moss-trade-bot/agent_creds.json
```

### 3. Live Trading
Create a Realtime Bot and start the automated trading loop:
```bash
# Create Bot
python3 live_trade.py create-bot --creds ~/.moss-trade-bot/agent_creds.json ...

# Run Auto Trading
python3 live_runner.py \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --params-file /tmp/bot_params.json \
  --interval 15 \
  --log /tmp/bot_live.log
```

## Contributing

We welcome contributions! Whether it's adding new technical indicators, improving the evolution logic, or enhancing the backtest engine, your input is valuable.

## License

This project is licensed under the MIT-0 License.
