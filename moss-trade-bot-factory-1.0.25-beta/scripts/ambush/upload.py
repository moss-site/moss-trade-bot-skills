#!/usr/bin/env python3
"""
Ambush bot 上传：直接调 server 端 V2 create_realtime_bot (HMAC)，
**不**走 backtest verify 路径。

设计 §5.1 / §6.3：ambush 没有 K 线连续回放，平台 verify (`/backtest/verify`) 期望
DecisionParams + 单 symbol K 线 fingerprint，与 ambush 双通道参数 + symbol="*"
不兼容。8 标杆 sanity check 在 skill 端本地完成，server 端无需另一次 verify。

流程：
  1. 读 ambush_params.json (propose.py 输出) + ambush_backtest_result.json
  2. 转 V2 wire 格式 (decimal → string, see trading_client._ambush_params_for_wire)
  3. 调 client.create_realtime_bot(strategy_type="ambush", ambush_params=...)
  4. 平台 ambush 注册器 (cmd/server/ambush.go RegisterAmbushBot) 自动订阅事件链
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from trading_client import TradingClient  # noqa: E402


def _load_creds(creds_path: str) -> dict:
    if not creds_path or not os.path.isfile(creds_path):
        raise SystemExit(
            f"--creds path required and must exist (got {creds_path!r}). "
            "Run live_trade.py bind first to save api_key/api_secret to agent_creds.json."
        )
    with open(creds_path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", required=True, help="ambush 参数 JSON（来自 propose.py）")
    p.add_argument("--backtest-result", default="",
                   help="本地回测结果 JSON（来自 backtest.py，仅作 console 摘要打印；不上传）")
    p.add_argument("--display-name", required=True, help="bot 名（中文）")
    p.add_argument("--display-name-en", default="", help="bot 名（英文，缺省 = display-name）")
    p.add_argument("--persona", required=True, help="人设（中文）")
    p.add_argument("--persona-en", default="", help="人设（英文，缺省 = persona）")
    p.add_argument("--description", required=True, help="描述（中文）")
    p.add_argument("--description-en", default="", help="描述（英文，缺省 = description）")
    p.add_argument("--platform-url", default="",
                   help="平台 URL，例如 https://moss-dev.moss.site；缺省读 creds.base_url")
    p.add_argument("--creds", required=True, help="agent_creds.json 路径（bind 后保存的 HMAC 凭证）")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要发送的请求体，不实际调用平台")
    args = p.parse_args()

    raw_params = json.loads(Path(args.params).read_text())
    if raw_params.get("strategy_type") != "ambush":
        print(
            f"[ambush-upload] ERROR: params strategy_type={raw_params.get('strategy_type')!r}, expected 'ambush'",
            file=sys.stderr,
        )
        sys.exit(1)

    # 本地回测结果只用于 console 摘要打印（让用户上线前再看一眼）；不上传。
    # 字段名与 scripts/ambush/backtest.py 输出的 summary 一致：
    #   triggered_count / win_rate / total_return_pct / max_drawdown_pct
    if args.backtest_result:
        try:
            bt = json.loads(Path(args.backtest_result).read_text())
            sn = bt.get("sanity_check", {}) or {}
            sm = bt.get("summary", {}) or {}
            print("[ambush-upload] local backtest summary:")
            print(f"  trades={sm.get('triggered_count', '?')}  "
                  f"win_rate={sm.get('win_rate', '?')}  "
                  f"total_return_pct={sm.get('total_return_pct', '?')}  "
                  f"max_drawdown_pct={sm.get('max_drawdown_pct', '?')}")
            if sn:
                print(f"  sanity_check: {sn.get('passed', '?')}/{sn.get('total', '?')} passed")
        except Exception as e:
            print(f"[ambush-upload] WARN: read backtest-result failed: {e}", file=sys.stderr)

    creds = _load_creds(args.creds)
    base_url = args.platform_url or creds.get("base_url", "")
    if not base_url:
        raise SystemExit(
            "platform base URL missing. Pass --platform-url or save base_url in agent_creds.json via live_trade.py bind."
        )

    if args.dry_run:
        # Build the wire body without sending — useful to debug shape.
        from trading_client import _ambush_params_for_wire
        body_preview = {
            "display_name": args.display_name,
            "persona": args.persona,
            "description": args.description,
            "strategy": {
                "params": {},
                "strategy_type": "ambush",
                "ambush_params": _ambush_params_for_wire(raw_params),
            },
        }
        print("[ambush-upload] DRY RUN — would POST to "
              f"{base_url.rstrip('/')}/api/v2/moss/agent/realtime/bots:")
        print(json.dumps(body_preview, indent=2, ensure_ascii=False))
        return

    client = TradingClient(
        api_key=creds.get("api_key", ""),
        api_secret=creds.get("api_secret", ""),
        base_url=base_url,
    )
    print(f"[ambush-upload] creating ambush bot at {base_url} ...")
    resp = client.create_realtime_bot(
        display_name=args.display_name,
        persona=args.persona,
        description=args.description,
        strategy_params={},  # ambush 不用 DecisionParams; server 据 strategy_type 分流
        display_name_i18n={"zh": args.display_name, "en": args.display_name_en or args.display_name},
        persona_i18n={"zh": args.persona, "en": args.persona_en or args.persona},
        description_i18n={"zh": args.description, "en": args.description_en or args.description},
        strategy_type="ambush",
        ambush_params=raw_params,
    )

    # Server returns 201 with the created bot resource on success, or
    # {"code": "...", "message": "..."} on rejection. Surface both clearly.
    if isinstance(resp, dict) and resp.get("code") and not resp.get("bot_id"):
        print(f"[ambush-upload] platform rejected: {resp}", file=sys.stderr)
        sys.exit(1)

    bot_id = resp.get("bot_id") or resp.get("agent_id") or "<unknown>"
    print(f"[ambush-upload] created ambush bot bot_id={bot_id}")
    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
