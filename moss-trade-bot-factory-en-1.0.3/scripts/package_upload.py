#!/usr/bin/env python3
"""Package backtest results into an upload bundle for platform verification.

Usage:
    python package_upload.py \
        --bot-name-zh "利弗莫尔" \
        --bot-name-en "Livermore" \
        --bot-personality-zh "趋势投机者" \
        --bot-personality-en "Trend Speculator" \
        --bot-description-zh "利弗莫尔风格的 BTC 趋势投机机器人。" \
        --bot-description-en "Livermore-style BTC trend speculation bot." \
        --params-file bot_params.json \
        --fingerprint-file fingerprint.json \
        --result-file backtest_result.json \
        --output upload_package.json

    # 显式上传并轮询（推荐显式给 --platform-url / --creds）
    python package_upload.py ... \
        --platform-url https://ai.moss.site \
        --creds ~/.moss-trade-bot/agent_creds.json
    # --platform-url should be site origin only, e.g. https://ai.moss.site

    # 进化回测上传（必须带 evolution_log，平台才会做分段 stitched 回放，否则退化成单参回放，和本地结果对不上）：
    python package_upload.py ... \
        --params-file /tmp/bot_params.json \
        --result-file /tmp/evolve_result_final.json \
        --evolution-log-file /tmp/evolve_result_final.json
    # 若 result-file 已是 run_evolve_backtest 输出（含 evolution_log），可不传 --evolution-log-file，脚本会从 result 里自动带出。
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.decision import DecisionParams
from text_i18n import default_text, validate_bilingual_text

PLATFORM_URL_HELP = "Platform site origin only, e.g. https://ai.moss.site. The client appends API paths automatically."

# 数据源必须与本地一致：Skill 仅使用 Binance USDT-M 期货 (binanceusdm)。
# 上传时默认原样使用 fingerprint 的 exchange，不再映射为 binance，避免平台用现货数据校验导致指纹/结果不一致。
# 若平台仅接受 binance 且将 binance 视为期货，可传 --exchange binance 覆盖。


def _materialize_params(raw_params):
    raw_params = raw_params or {}
    if not isinstance(raw_params, dict):
        return DecisionParams().to_dict()
    clean = {k: v for k, v in raw_params.items() if v is not None}
    return DecisionParams.from_dict(clean).to_dict()
def main():
    parser = argparse.ArgumentParser(description="Package upload bundle")
    parser.add_argument("--bot-name", default="")
    parser.add_argument("--bot-name-zh", required=True)
    parser.add_argument("--bot-name-en", required=True)
    parser.add_argument("--bot-personality", default="")
    parser.add_argument("--bot-personality-zh", required=True)
    parser.add_argument("--bot-personality-en", required=True)
    parser.add_argument("--bot-description", default="")
    parser.add_argument("--bot-description-zh", required=True)
    parser.add_argument("--bot-description-en", required=True)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--fingerprint-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--evolution-log-file", default=None)
    parser.add_argument("--evolution-config", default=None, help="JSON string for evolution config")
    parser.add_argument("--output", default="upload_package.json")
    parser.add_argument("--platform-url", default=None, help=PLATFORM_URL_HELP + " Otherwise reuse base_url stored in agent_creds.json.")
    parser.add_argument("--creds", default=None,
                        help="Path to agent_creds.json (bind 后保存的 api_key/api_secret/base_url)。上传验证必须显式提供")
    parser.add_argument("--exchange", default=None,
                        help="Override data_fingerprint.exchange. Default: use fingerprint's exchange (binanceusdm). Set to binance only if platform requires it.")
    args = parser.parse_args()

    with open(args.params_file) as f:
        params = json.load(f)
    with open(args.fingerprint_file) as f:
        fingerprint = json.load(f)
    with open(args.result_file) as f:
        result_data = json.load(f)

    # 平台 verifier 根据 evolution_log 是否为空决定：空则单参数普通回放，非空则分段 stitched evolve 回放。
    # 上传进化回测时必须带上 evolution_log，否则平台用 bot.params 单参回放，和本地“分段进化”结果不是同一类回测，必然对不上。
    evolution_log = []
    if args.evolution_log_file:
        with open(args.evolution_log_file) as f:
            evo_data = json.load(f)
        evolution_log = evo_data.get("evolution_log", evo_data) if isinstance(evo_data, dict) else evo_data
        if not isinstance(evolution_log, list):
            evolution_log = []
    if not evolution_log and isinstance(result_data, dict) and result_data.get("evolution_log"):
        evolution_log = result_data["evolution_log"]

    evolution_config = None
    if args.evolution_config:
        evolution_config = json.loads(args.evolution_config)

    params = _materialize_params(params)

    def _to_rfc3339(ts: str) -> str:
        if not ts:
            return ""
        ts = str(ts).strip()
        if ts.endswith("Z") and "T" in ts:
            return ts
        normalized = ts.replace(" ", "T")
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        elif re.match(r".*[+-]\d{2}:\d{2}$", normalized) is None:
            normalized += "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _json_safe_float(value, *, positive_inf=999999.0) -> float:
        value = float(value)
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return positive_inf if value > 0 else 0.0
        return value

    def _normalize_evolution_log(entries):
        normalized = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if isinstance(item.get("params_used"), dict):
                item["params_used"] = _materialize_params(item["params_used"])
            time_range = item.get("time_range")
            if isinstance(time_range, list):
                item["time_range"] = [_to_rfc3339(v) for v in time_range[:2]]
            seg = item.get("segment_result")
            if isinstance(seg, dict):
                seg_copy = dict(seg)
                for field in ("total_return", "win_rate", "avg_win_pct", "avg_loss_pct"):
                    if field in seg_copy:
                        seg_copy[field] = _json_safe_float(seg_copy[field])
                if "total_trades" in seg_copy:
                    seg_copy["total_trades"] = int(seg_copy["total_trades"])
                if "blowup_count" in seg_copy:
                    seg_copy["blowup_count"] = int(seg_copy["blowup_count"])
                item["segment_result"] = seg_copy
            normalized.append(item)
        return normalized

    backtest_result = result_data.get("backtest_result", result_data)
    trades = result_data.get("trades", [])
    for t in trades:
        if "entry_time" in t:
            t["entry_time"] = _to_rfc3339(t["entry_time"])
        if "exit_time" in t:
            t["exit_time"] = _to_rfc3339(t["exit_time"])
    evolution_log = _normalize_evolution_log(evolution_log)

    try:
        name_i18n = validate_bilingual_text(
            "bot.name_i18n",
            {"zh": args.bot_name_zh, "en": args.bot_name_en},
            64,
        )
        personality_i18n = validate_bilingual_text(
            "bot.personality_i18n",
            {"zh": args.bot_personality_zh, "en": args.bot_personality_en},
            64,
        )
        description_i18n = validate_bilingual_text(
            "bot.description_i18n",
            {"zh": args.bot_description_zh, "en": args.bot_description_en},
            280,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    package = {
        "version": "1.0",
        "bot": {
            "name": default_text(name_i18n),
            "name_i18n": name_i18n,
            "personality": default_text(personality_i18n),
            "personality_i18n": personality_i18n,
            "description": default_text(description_i18n),
            "description_i18n": description_i18n,
            "params": params,
            "evolution_config": evolution_config,
        },
        "data_fingerprint": {
            "symbol": fingerprint["symbol"],
            "timeframe": fingerprint["timeframe"],
            "exchange": args.exchange or fingerprint.get("exchange", "binanceusdm"),
            "start": _to_rfc3339(fingerprint["start"]),
            "end": _to_rfc3339(fingerprint["end"]),
            "bars": fingerprint["bars"],
            "first_close": fingerprint["first_close"],
            "last_close": fingerprint["last_close"],
            "checksum": fingerprint.get("checksum", ""),
        },
        "backtest_result": {
            "total_return": _json_safe_float(backtest_result["total_return"]),
            "sharpe_ratio": _json_safe_float(backtest_result["sharpe_ratio"]),
            "max_drawdown": _json_safe_float(backtest_result["max_drawdown"]),
            "win_rate": _json_safe_float(backtest_result["win_rate"]),
            "profit_factor": _json_safe_float(backtest_result["profit_factor"]),
            "total_trades": int(backtest_result["total_trades"]),
            "blowup_count": int(backtest_result.get("blowup_count", 0)),
        },
        "evolution_log": evolution_log,
        "trades": trades[:500],
    }

    with open(args.output, "w") as f:
        json.dump(package, f, indent=2, ensure_ascii=False)
    print(f"Upload package saved to {args.output}")

    if args.platform_url or args.creds:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from trading_client import TradingClient

            if not args.creds or not os.path.isfile(args.creds):
                print(
                    "Error: --creds required for platform upload. Path to agent_creds.json (from bind).",
                    file=sys.stderr,
                )
                sys.exit(1)
            with open(args.creds) as f:
                creds = json.load(f)
            upload_base_url = args.platform_url or creds.get("base_url", "")
            if not upload_base_url:
                print(
                    "Error: platform base URL missing. Pass --platform-url or save base_url in agent_creds.json via live_trade.py bind.",
                    file=sys.stderr,
                )
                sys.exit(1)
            api_key = creds.get("api_key", "")
            api_secret = creds.get("api_secret", "")
            client = TradingClient(api_key=api_key, api_secret=api_secret, base_url=upload_base_url)
            print(f"\nSubmitting to {upload_base_url}...")
            result = client.verify_backtest_and_wait(package)
            print(f"Platform response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Upload failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
