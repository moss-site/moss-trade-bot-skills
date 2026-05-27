#!/usr/bin/env python3
"""
Ambush bot 上传：支持两种模式。

## 模式 1 (默认): create-realtime-bot
直接调 server 端 V2 create_realtime_bot (HMAC)，不走 backtest verify 路径。

  1. 读 ambush_params.json (propose.py 输出) + ambush_backtest_result.json
  2. 转 V2 wire 格式 (decimal → string, see trading_client._ambush_params_for_wire)
  3. 调 client.create_realtime_bot(strategy_type="ambush", ambush_params=...)
  4. 平台 ambush 注册器 (cmd/server/ambush.go RegisterAmbushBot) 自动订阅事件链

## 模式 2 (--ambush-verify): chained backtest verify-job
本地运行 ChainedHarness，计算 fingerprint，POST 到 /api/v1/moss/agent/backtest/verify，
server 端重新验算 fingerprint 并入队 strategy_type='ambush' 的 verify job。

  1. 计算 dataset SHA-256 (data_cache/ambush/{events,features,klines})
  2. 计算 canonical fingerprint (params + initial_capital + harness_version + dataset_sha256)
  3. 本地运行 ChainedHarness 得到 local_result
  4. POST payload 到 server；server 验证 fingerprint 后入队 verify job
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from trading_client import TradingClient  # noqa: E402


_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache" / "ambush"


def _load_creds(creds_path: str) -> dict:
    if not creds_path or not os.path.isfile(creds_path):
        raise SystemExit(
            f"--creds path required and must exist (got {creds_path!r}). "
            "Run live_trade.py bind first to save api_key/api_secret to agent_creds.json."
        )
    with open(creds_path) as f:
        return json.load(f)


def upload_ambush_verify_job(
    creds: dict,
    base_url: str,
    raw_params: dict,
    data_dir: Path,
    user_uuid: str,
    display_name: str = "",
    display_name_en: str = "",
    persona: str = "",
    persona_en: str = "",
    description: str = "",
    description_en: str = "",
    dry_run: bool = False,
) -> tuple[int, dict]:
    """POST /backtest/verify-job with the ambush payload shape.

    Workflow (Task 3.3):
      1. Load dataset + compute dataset SHA-256.
      2. Compute canonical fingerprint (params + initial_capital + harness_version).
      3. Run the ChainedHarness locally to produce local_result metrics.
      4. Build and POST the full payload to /api/v1/moss/agent/backtest/verify-job.

    Returns (status_code, response_dict).
    """
    from ambush.chained_harness import AmbushBotParams, ChainedHarness, load_dataset
    from ambush.fingerprint import canonical, dataset_sha256, CURRENT_HARNESS_VERSION

    params = AmbushBotParams.from_dict(raw_params)

    # Initial capital from params or default $10000.
    initial_capital_str = str(raw_params.get("initial_capital", "10000") or "10000")
    initial_capital = Decimal(initial_capital_str)

    print(f"[ambush-verify] computing dataset SHA from {data_dir} ...")
    ds_sha = dataset_sha256(data_dir)
    print(f"[ambush-verify] dataset SHA: {ds_sha}")

    fp = canonical(
        strategy_type="ambush",
        harness_version=CURRENT_HARNESS_VERSION,
        dataset_sha256=ds_sha,
        initial_capital=initial_capital,
        params=params,
    )
    print(f"[ambush-verify] local fingerprint: {fp}")

    print(f"[ambush-verify] running chained harness locally ...")
    dataset = load_dataset(data_dir)
    result = ChainedHarness(params).run(dataset, initial_capital)
    print(
        f"[ambush-verify] harness done: events_total={result.events_total} "
        f"events_traded={result.events_traded} events_skipped={result.events_skipped} "
        f"final_wallet={result.final_wallet}"
    )

    # Build bot.* metadata using the same i18n shape the majors verify-job
    # uses (server's backtest.UploadBot expects name_i18n / personality_i18n /
    # description_i18n, NOT flat name_zh / persona_zh — those just get dropped
    # at JSON unmarshal time and the backtest-agent row ends up with empty
    # display_name/persona/description, leaving "Agent agt_xxx" + "暂无描述"
    # on the leaderboard).
    #
    # Defaults are sensible placeholders when the caller doesn't supply args
    # (so a bare `upload.py --ambush-verify` still gives the backtest agent a
    # readable identity rather than blanks).
    name_zh = display_name or "Ambush 异动币回测"
    name_en = display_name_en or display_name or "Ambush backtest"
    persona_zh = persona or "异动币策略 chained-backtest"
    persona_en = persona_en or persona or "Ambush chained-backtest"
    desc_zh = description or "基于历史异动事件链式回测;不绑定固定币种,按 direction/position_pct/leverage/stop/trailing 等参数复盘 216 个事件。"
    desc_en = description_en or description or "Chained-backtest replay over historical ambush events; not bound to one symbol, walks 216 events with direction/position_pct/leverage/stop/trailing."

    payload = {
        "version": "1.0",
        "bot": {
            "name": name_zh,
            "name_i18n": {"zh": name_zh, "en": name_en},
            "personality": persona_zh,
            "personality_i18n": {"zh": persona_zh, "en": persona_en},
            "description": desc_zh,
            "description_i18n": {"zh": desc_zh, "en": desc_en},
            "params": {},
        },
        # Legacy majors-shaped fields kept for schema compatibility (omitempty on server).
        "data_fingerprint": {
            "symbol": "*", "timeframe": "15m", "exchange": "hyperliquid",
            "bars": 0, "first_bar_open_time": "2025-01-01T00:00:00Z",
            "last_bar_close_time": "2026-01-01T00:00:00Z", "fingerprint_hex": "",
        },
        "backtest_result": {
            "version": "1.0", "total_return_pct": 0, "sharpe": 0,
            "max_drawdown_pct": 0, "win_rate": 0, "total_trades": 0,
        },
        # Ambush-specific fields.
        "strategy_type": "ambush",
        "ambush_params": params.to_dict(),
        "initial_capital": str(initial_capital),
        "harness_version": CURRENT_HARNESS_VERSION,
        "dataset_sha256": ds_sha,
        "local_fingerprint": fp,
        "local_result": {
            "final_wallet":     str(result.final_wallet.quantize(Decimal("0.01"))),  # cents, matches StringFixed(2)
            "events_total":     result.events_total,
            "events_traded":    result.events_traded,
            "events_skipped":   result.events_skipped,
            "total_trades":     result.total_trades,
            "win_rate":         f"{result.win_rate:.4f}",
            "sharpe":           f"{result.sharpe:.4f}",
            "max_drawdown_pct": f"{result.max_drawdown_pct:.4f}",  # raw signed pct (negative for drawdowns)
        },
    }

    if dry_run:
        print("[ambush-verify] DRY RUN — would POST:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0, payload

    # Route: agentBacktest.POST("/backtest/verify") under /api/v1/moss/agent
    # Uses HMAC auth via TradingClient (same as create_realtime_bot).
    client = TradingClient(
        api_key=creds.get("api_key", ""),
        api_secret=creds.get("api_secret", ""),
        base_url=base_url,
    )
    print(f"[ambush-verify] POSTing /backtest/verify (HMAC) to {base_url} ...")
    try:
        body = client._request(
            method="POST",
            path="/backtest/verify",
            body=payload,
            custom_prefix="/api/v1/moss/agent",
        )
        status = 200
    except Exception as exc:
        # TradingClient raises urllib.error.HTTPError on non-2xx responses.
        import urllib.error
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = json.loads(exc.read().decode())
            except Exception:
                body = {"raw": str(exc)}
            status = exc.code
        else:
            raise
    return status, body


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", required=True, help="ambush 参数 JSON（来自 propose.py）")
    p.add_argument("--backtest-result", default="",
                   help="本地回测结果 JSON（来自 backtest.py，仅作 console 摘要打印；不上传）")
    p.add_argument("--display-name", default="",
                   help="bot 名（中文，create-bot 路径必填；--ambush-verify 路径忽略）")
    p.add_argument("--display-name-en", default="", help="bot 名（英文，缺省 = display-name）")
    p.add_argument("--persona", default="", help="人设（中文，create-bot 路径必填）")
    p.add_argument("--persona-en", default="", help="人设（英文，缺省 = persona）")
    p.add_argument("--description", default="", help="描述（中文，create-bot 路径必填）")
    p.add_argument("--description-en", default="", help="描述（英文，缺省 = description）")
    p.add_argument("--platform-url", default="",
                   help="平台 URL，例如 https://moss-dev.moss.site；缺省读 creds.base_url")
    p.add_argument("--creds", required=True, help="agent_creds.json 路径（bind 后保存的 HMAC 凭证）")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要发送的请求体，不实际调用平台")
    # Ambush verify-job mode (Task 3.3).
    p.add_argument(
        "--ambush-verify", action="store_true",
        help=(
            "Run the chained harness locally, compute fingerprint, and POST to "
            "/api/v1/moss/agent/backtest/verify-job instead of creating a realtime bot. "
            "Requires --params (ambush params JSON) and --data-dir (or uses default "
            "data_cache/ambush). Server must have Stage 2 (verify dispatcher) deployed."
        ),
    )
    p.add_argument(
        "--data-dir", default="",
        help="Path to data_cache/ambush directory (for --ambush-verify). "
             "Defaults to <skill_root>/data_cache/ambush.",
    )
    p.add_argument(
        "--user-uuid", default="",
        help="User UUID for /backtest/verify-job?user_uuid= (for --ambush-verify). "
             "If omitted, reads user_uuid from creds file.",
    )
    args = p.parse_args()

    raw_params = json.loads(Path(args.params).read_text())

    # --- Ambush verify-job path (Task 3.3) ---
    if args.ambush_verify:
        if raw_params.get("strategy_type") != "ambush":
            print(
                f"[ambush-verify] ERROR: params strategy_type={raw_params.get('strategy_type')!r}, expected 'ambush'",
                file=sys.stderr,
            )
            sys.exit(1)
        creds = _load_creds(args.creds)
        base_url = args.platform_url or creds.get("base_url", "")
        if not base_url:
            raise SystemExit(
                "platform base URL missing. Pass --platform-url or save base_url in agent_creds.json."
            )
        user_uuid = args.user_uuid or creds.get("user_uuid", "")
        data_dir = Path(args.data_dir) if args.data_dir else _DEFAULT_DATA_DIR
        status, body = upload_ambush_verify_job(
            creds=creds,
            base_url=base_url,
            raw_params=raw_params,
            data_dir=data_dir,
            user_uuid=user_uuid,
            display_name=args.display_name,
            display_name_en=args.display_name_en,
            persona=args.persona,
            persona_en=args.persona_en,
            description=args.description,
            description_en=args.description_en,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"[ambush-verify] HTTP {status}")
            print(json.dumps(body, indent=2, ensure_ascii=False))
            if status not in (200, 201):
                sys.exit(1)
        return

    # --- Original create-realtime-bot path ---
    if not args.display_name:
        raise SystemExit("--display-name is required for the create-bot path (or use --ambush-verify)")
    if not args.persona:
        raise SystemExit("--persona is required for the create-bot path (or use --ambush-verify)")
    if not args.description:
        raise SystemExit("--description is required for the create-bot path (or use --ambush-verify)")

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
            sm = bt.get("summary", {}) or {}
            print("[ambush-upload] local backtest summary:")
            print(f"  trades={sm.get('triggered_count', '?')}  "
                  f"win_rate={sm.get('win_rate', '?')}  "
                  f"total_return_pct={sm.get('total_return_pct', '?')}  "
                  f"max_drawdown_pct={sm.get('max_drawdown_pct', '?')}")
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
    # Persist bot_id back into the creds file (parity with live_trade.py
    # create-bot) so downstream tooling — live_runner, exit_watcher — can use
    # the creds directly without a manual DB lookup. Only bot_id is written;
    # base_url is left untouched (we never clobber it with --platform-url).
    if bot_id and bot_id != "<unknown>":
        try:
            creds["bot_id"] = bot_id
            with open(args.creds, "w") as f:
                json.dump(creds, f, indent=2, ensure_ascii=False)
            print(f"[ambush-upload] bot_id saved to {args.creds}")
        except OSError as e:
            print(f"[ambush-upload] WARN: failed to write bot_id to {args.creds}: {e}", file=sys.stderr)
    print(json.dumps(resp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
