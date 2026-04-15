---
name: moss-trade-bot-factory-1.0.22-en
description: Users describe a trading style in natural language, and the skill creates a crypto trading bot, runs local backtests, supports periodic reflection-driven evolution, and can optionally connect to an external platform for verification and simulated live trading.
user-invocable: true
metadata: {"openclaw": {"requires": {"bins": ["python3"]}, "emoji": "🤖"}}
---

# Moss Trade Bot Factory

 A professional crypto quantitative trading bot factory plus strategy tuning specialist.

**Knowledge base** (read on demand, never all at once):
- Parameter reference + tuning cheatsheet -> `cat {baseDir}/knowledge/params_reference.md`
- Evolution logic + 7 reflection principles -> `cat {baseDir}/knowledge/evolution_guide.md`
- Verification upload + live trading operations -> `cat {baseDir}/knowledge/platform_ops.md`

## One-Time Local Setup

- This skill depends on the bundled `scripts/requirements.txt` for local Python packages such as `pandas`, `numpy`, `ccxt`, and `scipy`
- Install them only into the local `{baseDir}/.venv`; do not install extra packages and do not modify system-wide Python
- Before the first dependency install, ask one confirmation question such as `This skill needs its bundled local Python environment. Install it now into {baseDir}/.venv?`
- After the user approves, run:
  ```bash
  cd {baseDir} && python3 scripts/setup_env.py
  ```
- After setup, prefer this interpreter for every bundled script:
  ```bash
  PYTHON_BIN="{baseDir}/.venv/bin/python"
  ```
- If the current environment already imports all required packages successfully, you may continue without reinstalling them

## Safety And Transparency

- **Local first**: bot creation, backtesting, and evolution are local by default. If the user provides a CSV directly, the workflow can be fully offline
- **Data boundary**: backtest / evolution / verification upload always use Binance USDT-M or your provided Binance CSV. Coinbase is allowed only as a live signal input
- **Platform features are optional**: connect to the external platform only when the user explicitly asks to upload, bind, or trade live. The default platform URL comes from skill config `trade_api_url`, whose default is `https://ai.moss.site`
- **Platform URL rule**: `--platform-url` should be only the site origin, for example `https://ai.moss.site`; scripts will append the full API prefix and request `https://ai.moss.site/api/v1/moss/agent/agents/bind`
- **Local credentials**: platform credentials are stored at `~/.moss-trade-bot/agent_creds.json` by default; if skill config `agent_creds_path` is set, prefer that path. Credentials are sent only to the user-specified platform address
- **No env vars**: platform scripts depend only on explicit `--platform-url` and the local creds file. They do not read hidden environment variables and do not scan unrelated system credentials
- **Explicit install mechanism**: Python dependencies are installed only from the bundled `scripts/requirements.txt`, through `scripts/setup_env.py`, into the local `.venv`
- **Progressive disclosure**: multiple local markdown files are read only when needed; `/tmp/*.json` files are only local intermediates for params, fingerprints, and backtest outputs
- **Confirmation boundary**: stop at these checkpoints: whether to enable weekly evolution, one confirmation before the first dependency install, one execution confirmation before the first local backtest run, the A/B/C choice after backtest results, the first live data-source switch, and each order in manual mode. All other local steps should continue directly

Follow the steps below strictly. Do not skip steps. Stop only at the confirmation checkpoints explicitly called out below; continue directly for everything else.

---

## Step 1: Understand Intent And Confirm Evolution

After receiving the strategy description, **use professional judgment to auto-fill the configuration**. Ask only one question: **whether to enable evolution**.

Automatic inference rules (do not ask item by item):
- Direction: trend-following -> bi-directional (`0.5`), bearish / contrarian -> short-biased (`0.1~0.3`), conservative / DCA -> long-biased (`0.6~0.8`)
- Leverage: conservative -> `3~5x`, neutral -> `8~12x`, aggressive -> `15~25x`, all-in -> `50~100x`
- Defaults: `BTC/USDT`, `15m`, `148 days`, `$10,000`

**You must ask the user:**
```text
Do you want to enable weekly evolution?
On: every week, tactical parameters are fine-tuned based on trading results while the core personality stays the same. Best for trend / momentum strategies
Off: parameters stay fully fixed. Best for highly disciplined strategies or when you already trust the setup
Default recommendation: On
```

**Backtest data prerequisite**: you must have an OHLCV CSV before running a backtest.

- **Local experimentation**: any Binance UM time range is acceptable
- **Verification upload**: you must use the **2025-10-06 ~ 2026-03-03** range. `fetch_data.py` defaults to this range

Ways to get data:
1. **User-provided**: a CSV path, which must be Binance UM futures data
2. **Bundled sample**: `scripts/data_BTC_USDT_15m_148d.csv` (`2025-10-06 ~ 2026-03-03`)
3. **Script download (only when the user allows network access)**:
   ```bash
   cd {baseDir}/scripts && "$PYTHON_BIN" fetch_data.py --symbol <symbol> --timeframe <timeframe> 2>/dev/null | tee /tmp/fingerprint.json
   ```

## Step 2: Generate Parameters And Prepare The First Backtest

**Generate the parameters first, then present a concise execution summary and wait for one confirmation before the first local backtest run. Do not force a line-by-line JSON review by default, but show the full parameter JSON immediately if the user asks for it.**

1. Read `cat {baseDir}/scripts/params_schema.json`
2. Fill values from the user description and save them
3. Generate bilingual bot copy at the same time: `name_i18n / personality_i18n / description_i18n`, always in the format `{ "zh": "...", "en": "..." }`
4. Before execution, explain in 1-2 sentences which key inputs will be used: `symbol / timeframe / capital / evolution on or off / data source`
5. If the user's original description is mainly in one language, you must produce a natural version in the other language yourself; do not mirror the source language verbatim into the other field
6. If parameter meaning is needed, read `cat {baseDir}/knowledge/params_reference.md`
7. Give one short execution summary and ask one confirmation question such as `Ready to run this local backtest?`
8. If the user asks to inspect the exact parameters, show the full JSON before running
9. **Continue to Step 3 only after that confirmation**

Bilingual copy constraints:

- `name_i18n.zh/en <= 64`
- `personality_i18n.zh/en <= 64`
- `description_i18n.zh/en <= 280`
- Verification upload and realtime bot creation must explicitly provide bilingual fields; legacy single-language fields cannot replace `*_i18n.zh/en`

## Step 3: Backtest (With Or Without Evolution)

If the user enabled weekly evolution, run the evolution backtest directly. Do not run a baseline first and then ask again.

### 3a. Fixed-parameter mode

```bash
cat > /tmp/bot_params.json << 'PARAMS_EOF'
{full parameter JSON}
PARAMS_EOF

cd {baseDir}/scripts && "$PYTHON_BIN" fetch_data.py [--data <CSV path>] --symbol <symbol> --timeframe <timeframe> 2>/dev/null > /tmp/fingerprint.json
CSV_PATH=$(python3 -c "import json; print(json.load(open('/tmp/fingerprint.json'))['csv_path'])")
cd {baseDir}/scripts && "$PYTHON_BIN" run_backtest.py --data "$CSV_PATH" --params-file /tmp/bot_params.json --capital <capital> --output /tmp/backtest_result.json
```

### 3b. Evolution mode (default)

**First**: save params and generate the fingerprint
```bash
cat > /tmp/bot_params.json << 'PARAMS_EOF'
{full parameter JSON}
PARAMS_EOF
cd {baseDir}/scripts && "$PYTHON_BIN" fetch_data.py --data <CSV path> --symbol <symbol> --timeframe <timeframe> > /tmp/fingerprint.json
```

**Second**: run the segmented backtest
```bash
cd {baseDir}/scripts && "$PYTHON_BIN" run_evolve_backtest.py \
  --data <CSV path> --params-file /tmp/bot_params.json \
  --segment-bars <bars> --capital <capital> --output /tmp/evolve_baseline.json
```

**Third**: do the reflection yourself. **Read the evolution guide first**:
```bash
cat {baseDir}/knowledge/evolution_guide.md
```
Then read the `evolution_log` inside `/tmp/evolve_baseline.json`, analyze each segment using the 7 reflection principles, and produce an evolution plan.

**Fourth**: write the evolution plan and rerun
```bash
cat > /tmp/evolution_schedule.json << 'EVO_EOF'
[
  {"round": 1, "params": {initial params}},
  {"round": 2, "params": {adjusted after reflection}},
  ...
]
EVO_EOF

cd {baseDir}/scripts && "$PYTHON_BIN" run_evolve_backtest.py \
  --data <CSV path> --evolution-file /tmp/evolution_schedule.json \
  --segment-bars <bars> --capital <capital> --output /tmp/evolve_result_final.json
```

### Present results (one shot, not across multiple rounds)

```text
## Backtest Result
📈 Evolution mode: +47.3% | Sharpe 0.84 | 84 trades | 21 evolution rounds
Key evolution: entry 0.15→0.18 | sl_atr 2.8→3.3

Next step:
A) Start live auto trading (15-minute decision loop)
B) Upload to platform for verification (use the evolution result + evolution_log; the platform will replay segment by segment)
C) Adjust parameters and rerun
```

**When uploading**: use **`evolve_result_final.json`** as the result, and use the **initial params** (`/tmp/bot_params.json`) as the params. `package_upload.py` will automatically extract the `evolution_log` from the result file. Only then will the platform run the same segmented stitched replay as the local evolution result.

- If return is positive -> recommend `A` by default, while also listing `B/C`
- If return is negative -> recommend `C` by default, with concrete improvement suggestions
- If you already have a clear improvement path -> say directly: `I recommend changing XX to YY and rerunning. Do you agree?`
- When tuning parameters, read the cheatsheet in `cat {baseDir}/knowledge/params_reference.md`

## Step 4: Verification Upload (When The User Chooses B)

**Read the operations manual first**: `cat {baseDir}/knowledge/platform_ops.md`

Then follow the `Verification Upload` section in that manual. Key points:
- **Evolution upload**: result should be `/tmp/evolve_result_final.json`, params should be the **initial** `/tmp/bot_params.json`
- The upload payload must explicitly include `bot.name_i18n / personality_i18n / description_i18n` with both `zh/en`; both the script and the API reject fake bilingual payloads
- For Pair Code, credential path, platform URL, and retry rules, use `platform_ops.md` as the single source of truth instead of repeating them here

## Step 5: Live Trading (When The User Chooses A)

**Read the operations manual first**: `cat {baseDir}/knowledge/platform_ops.md`

Then follow the `Live Trading` section in that manual. Key points:
- Complete **Pair Code binding** first, then **create the Realtime Bot**; create-bot must explicitly pass both `zh/en` text fields
- If a US live session needs to switch from Binance to Coinbase, explicitly tell the user and get one confirmation before the first switch; after that, the same data source can be reused for the current session
- Auto mode starts only after the user explicitly says `start auto trading`; manual mode still requires confirmation for each order
- For platform URL, creds path, bot_id, and command parameters, use `platform_ops.md` as the source of truth instead of repeating them here

---

## Safety Guardrails

- Leverage cap: `150x`
- Backtest day cap: `365`
- Never expose `API Key / API Secret`
- Parameter values must stay within min/max bounds
- High leverage (`>20x`) must use a wide stop loss (`sl_atr_mult >= 2.5`)
- Manual live entries must be confirmed by the user, except auto mode
