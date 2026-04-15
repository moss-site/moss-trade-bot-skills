# Platform Operations Manual (Verification Upload + Live Trading)

## General Prerequisites

- Python setup: prefer the bundled local virtualenv at `{baseDir}/.venv`. If it does not exist yet, install dependencies first with `cd {baseDir} && python3 scripts/setup_env.py`, then use `PYTHON_BIN="{baseDir}/.venv/bin/python"` for the commands below
- Credential path: prefer skill config `agent_creds_path`; if not configured, default to `~/.moss-trade-bot/agent_creds.json` (**do not use `/tmp`**, it will be lost after restart)
- Platform URL: prefer skill config `trade_api_url`; the default is `https://ai.moss.site`. You may also pass it explicitly with `--platform-url`. After the first bind, the value is stored in `agent_creds.json` under `base_url` for reuse
- Authentication: HMAC signing (`api_key + api_secret`)
- The examples below all use the default path. If skill config already provides `agent_creds_path`, replace the creds path in all examples consistently
- `--platform-url` should contain only the site origin, for example `https://ai.moss.site`. The script will automatically build `https://ai.moss.site/api/v1/moss/agent/agents/bind`

## Dependency Declaration And Harmlessness

- Platform-related scripts depend only on two kinds of external input: the explicit platform URL (`--platform-url` or skill config `trade_api_url`) and the local `agent_creds.json` file
- Python packages are installed only from the bundled `scripts/requirements.txt`, through the bundled `scripts/setup_env.py`, into the local `.venv`
- `agent_creds.json` stores only the `api_key/api_secret` returned by bind, the later `bot_id`, and the optional `base_url`. It does not store system credentials or third-party secrets
- This is a local-file dependency, not an environment-variable dependency. The file is read only when the user explicitly enables upload, bind, or live trading
- Backtest results, CSVs, and parameter files are read locally only. Requests are sent to the user-specified platform address only when you explicitly run upload, bind, or live commands
- The local credentials created after bind can be used directly to submit and poll verification results

---

## Pair Code Binding (Required Before Upload Or Live Trading)

1. **Sign up / log in**: visit [Moss Trader](https://moss.site/agent)
2. **Get the Pair Code**: after login, the platform shows a **pair code** which the user copies
3. **Run bind**:
   ```bash
   mkdir -p ~/.moss-trade-bot
   cd {baseDir}/scripts && "$PYTHON_BIN" live_trade.py bind \
     --platform-url "https://ai.moss.site" \
     --pair-code "<pair_code>" \
     --name "<Bot Name>" --persona "<Style>" --description "<Strategy Description>" \
     --save ~/.moss-trade-bot/agent_creds.json
   ```
4. The response returns `binding_id`, `api_key`, and `api_secret` (**bind performs identity binding only, and does not create a live bot**). **`api_secret` is returned only once, so do not print it in the reply.** If `--save` is used, the same file also stores `base_url` for later reuse
5. **Before live trading, you must still create a Realtime Bot** (see `Create Realtime Bot` below). Only after the returned `bot_id` is written into the same creds file can you call account / positions / orders endpoints

---

## Verification Upload (Step 4)

### Data requirements

The platform validates against the **2025-10-06 ~ 2026-03-03** range on the server side. Both fingerprint and result must be based on that range. Local experimentation may use other ranges, but before upload you must rerun on this required range.

- Backtest / verification upload always use **Binance USDT-M futures**
- If the Binance API is unavailable, switch to your provided Binance CSV. Do not switch to Coinbase during backtest or upload
- Bot copy inside the upload package must now explicitly include bilingual fields: `name_i18n/personality_i18n/description_i18n = {zh, en}`
- The upload API strictly validates bilingual fields in the request body: if any `*_i18n.zh/en` is missing, the request is rejected even if legacy single-language fields are present
- If the Chinese copy contains Chinese characters, `package_upload.py` requires a natural English version. Do not mirror the Chinese text directly into `en`

### Must confirm before upload (all required)

1. The user has already completed bind and the creds file exists
2. The user explicitly says `upload`, `submit for verification`, or equivalent

### Important: verifier behavior

`evolution_log` / `--evolution-log-file` is optional at the API level. **If omitted = fixed-parameter mode** (the platform replays the whole range with `bot.params` only). **If provided = evolution mode** (the platform performs segmented stitched replay using the `evolution_log`, matching local `run_evolve_backtest`).

- Local `run_backtest.py` / `run_evolve_backtest.py` and the platform verifier now both use **cross-margin semantics**
- Opening a position consumes account `free_margin`
- Liquidation is triggered at the account level by `equity <= maintenance_margin_total`, not by “a single position losing its own margin”

- **Non-empty `evolution_log`**: the platform performs **segmented stitched replay** using each segment's `params_used`, and compares the result against the submitted `backtest_result`
- **Empty `evolution_log`**: the platform falls back to **single-parameter replay** across the full range, which is **not the same class of backtest** as local segmented evolution. Trade count and returns will not match

Therefore: **if this is an evolution backtest, the upload must include `evolution_log`**. Otherwise the platform will compare a fixed-parameter replay against a segmented-evolution local result.

### Evolution upload (recommended, same-class comparison as the platform)

Use the **`run_evolve_backtest` output** as the result and include its `evolution_log` from the same file. Use the **initial parameters** from before evolution as the params.

```bash
cd {baseDir}/scripts && "$PYTHON_BIN" package_upload.py \
  --bot-name-zh "<Chinese Name>" \
  --bot-name-en "<English Name>" \
  --bot-personality-zh "<Chinese Style Label>" \
  --bot-personality-en "<English Personality>" \
  --bot-description-zh "<Chinese Strategy Description, <=280 chars>" \
  --bot-description-en "<English Strategy Description, <=280 chars>" \
  --params-file /tmp/bot_params.json \
  --fingerprint-file /tmp/fingerprint.json \
  --result-file /tmp/evolve_result_final.json \
  --output /tmp/upload_package.json \
  --platform-url https://ai.moss.site \
  --creds ~/.moss-trade-bot/agent_creds.json
```

Note: `evolve_result_final.json` already includes `evolution_log`, and `package_upload.py` extracts it automatically. If you want to pass it explicitly, you may use `--evolution-log-file /tmp/evolve_result_final.json`.

Additional notes:

- `--bot-name / --bot-personality / --bot-description` are still kept as compatibility projections
- The new script also writes `bot.name_i18n / bot.personality_i18n / bot.description_i18n`
- If only Chinese is provided and no English translation is given, the script fails directly rather than mirroring Chinese into `en`
- Verification submission and polling require only the local `agent_creds.json` HMAC credentials

### Fixed-parameter upload (only when evolution was not used)

If you only ran `run_backtest` and did not run evolution, then use the `run_backtest` output as the result and upload without `evolution_log`. The platform then performs fixed-parameter replay.

### After packaging: submit and poll automatically (up to 120 seconds)

The command above already includes packaging. When `--platform-url` and `--creds` are provided, it also submits automatically and polls the result.

### Verification result handling

- **verified** -> passed. The platform automatically creates an Agent; tell the user the `bot_id`
- **rejected** -> do not ask the user. Analyze `mismatch_details` yourself:
  - precision issue (difference <1%) -> replace with `verified_result` and resubmit
  - fingerprint mismatch -> redownload data and regenerate the fingerprint
  - large difference (>10%) -> tell the user the platform replay engine differs significantly
  - auto-retry at most 2 times
- **failed** -> internal platform error, retry later

### Verification rules

- Hard fingerprint checks: candle-count difference <= 2%, first/last close difference <= 0.1%
- Checksum mismatch is warning-only
- Segmented-result tolerance: 2%, total-result tolerance: 1%

---

## Live Trading (Step 5)

### Prerequisite: bind + create Realtime Bot

- **Bind**: see `Pair Code Binding` above. This returns `binding_id`, `api_key`, and `api_secret`, which are saved into the creds file
- **Create Realtime Bot** (must be executed once before live trading):
  ```bash
  cd {baseDir}/scripts && "$PYTHON_BIN" live_trade.py create-bot \
    --creds ~/.moss-trade-bot/agent_creds.json \
    --platform-url "https://ai.moss.site" \
    --name "<Chinese Bot Name or Default Name>" \
    --name-zh "<Chinese Bot Name>" \
    --name-en "<English Bot Name>" \
    --persona "<Chinese Style Label or Default Style>" \
    --persona-zh "<Chinese Style Label>" \
    --persona-en "<English Persona>" \
    --description "<Chinese Strategy Description>" \
    --description-zh "<Chinese Strategy Description>" \
    --description-en "<English Strategy Description>" \
    --params-file /tmp/bot_params.json
  ```
  The script writes the returned `bot_id` into the same creds file. **If multiple realtime bots exist under the same binding**, account / positions / orders requests must include `X-BOT-ID`. This skill sends it automatically through the `bot_id` stored in the creds file. If only one active bot exists under the binding, the server may infer it

**Unbind semantics**: `unbind` only **deletes the current realtime bot** from list/public views. It does **not** revoke the binding credential itself. Full identity unbinding, if needed, is a platform-side operation

### Preflight checks

```bash
ls -la ~/.moss-trade-bot/agent_creds.json 2>/dev/null || true
# If needed, confirm whether the platform URL is already saved in local creds:
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".moss-trade-bot" / "agent_creds.json"
if p.exists():
    print(json.load(p.open()).get("base_url", ""))
PY
# The creds file should contain bot_id after create-bot has been run
```

### Run the bot automatically

```bash
cd {baseDir}/scripts && "$PYTHON_BIN" live_runner.py \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --platform-url "https://ai.moss.site" \
  --params-file /tmp/bot_params.json \
  --interval 15 \
  --log /tmp/bot_live.log
```

If a US user cannot access the Binance API, switch only the live auto-run signal source to Coinbase:

```bash
cd {baseDir}/scripts && "$PYTHON_BIN" live_runner.py \
  --creds ~/.moss-trade-bot/agent_creds.json \
  --platform-url "https://ai.moss.site" \
  --params-file /tmp/bot_params.json \
  --interval 15 \
  --data-source coinbase \
  --log /tmp/bot_live.log
```

Parameters:
- `--interval 15` -> one decision every 15 minutes, aligned with `15m` candles
- `--data-source coinbase` -> switches only the **live signal input** to Coinbase spot candles. It does **not** change the execution target, backtest data, fingerprint, or verification upload
- `--max-cycles 96` -> stop after 96 cycles (24 hours). `0` means unlimited
- `Ctrl+C` stops gracefully

Execution rules:
- If the context already makes it clear that the user is in the US, or the problem is already explicitly “US users cannot use the Binance API for live auto trading”, first tell the user you will switch to `--data-source coinbase`
- Before the first live data-source switch, obtain one user confirmation; after that, the same source can be reused for the rest of the current session
- Backtest, evolution replay, fingerprint, and verification upload must still use Binance USDT-M or your provided Binance CSV. Do not switch those to Coinbase because of US restrictions

### Manual trading

```bash
cd {baseDir}/scripts

# status
"$PYTHON_BIN" live_trade.py status --creds ~/.moss-trade-bot/agent_creds.json

# open long / short
"$PYTHON_BIN" live_trade.py open-long --creds ~/.moss-trade-bot/agent_creds.json --amount 1000 --leverage 10
"$PYTHON_BIN" live_trade.py open-short --creds ~/.moss-trade-bot/agent_creds.json --amount 1000 --leverage 10

# close position
"$PYTHON_BIN" live_trade.py close --creds ~/.moss-trade-bot/agent_creds.json --side LONG

# history
"$PYTHON_BIN" live_trade.py orders --creds ~/.moss-trade-bot/agent_creds.json
"$PYTHON_BIN" live_trade.py trades --creds ~/.moss-trade-bot/agent_creds.json
```

### Trading rules

- BTCUSDT perpetual only, market orders only
- Leverage `1-150x`
- Order amount = `free_margin × risk_per_trade × leverage`
- Check `free_margin` before opening a position
- On `STALE_MARK_PRICE`, wait a few seconds and retry
- Use `client_order_id` for idempotency, format: `{bot_name}-{timestamp}`

### Safety guardrails

**Manual mode**: report side / amount / leverage before every open order and wait for user confirmation  
**Auto mode**: if the user explicitly says `start auto trading`, that is authorization to start directly without per-order confirmation

General:
- Never print `api_secret` in the reply
- Before starting auto mode, make sure the user has seen the backtest result and understands the risk
- Surface errors to the user when they happen
