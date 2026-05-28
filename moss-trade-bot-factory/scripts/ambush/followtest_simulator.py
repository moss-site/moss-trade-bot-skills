#!/usr/bin/env python3
"""
Ambush copy-trading test simulator.

This is a skill-side QA loop. It does not rely on the backend realtime
schedule. In default mode every cycle it either:

  - injects a synthetic ambush event through _debug_publish, then manually
    opens a small position on the ambush bot; or
  - closes the current position.

With --always-position it keeps the bot invested: if a position exists, the
cycle makes a simulated decision to add or reduce. Reduce only closes part of
the current qty, so the bot remains in-position.

The resulting source orders/fills/events are intentionally real backend writes,
so follower/copy-trading clients can consume them through the normal Moss
follower/source-events/fills surfaces.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ambush.e2e_inject import RULE_PRESETS  # noqa: E402
from trading_client import TradingClient  # noqa: E402


LOG = logging.getLogger("ambush.followtest")


def _load_json(path: str) -> dict:
    with Path(path).expanduser().open() as f:
        return json.load(f)


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace(":", "").replace("-", "")


def _position_for_symbol(client: TradingClient, symbol: str) -> dict | None:
    target = _normalize_symbol(symbol)
    positions = client.get_positions()
    if isinstance(positions, dict):
        raise RuntimeError(f"get_positions failed: {positions}")
    for pos in positions or []:
        got = _normalize_symbol(pos.get("symbol", "") or symbol)
        qty = str(pos.get("qty", pos.get("net_qty", "0")) or "0").strip()
        try:
            is_open = abs(float(qty)) > 0
        except ValueError:
            is_open = qty not in ("", "0", "0.0")
        if got == target and is_open:
            return pos
    return None


def _position_qty(position: dict) -> Decimal:
    raw = str(position.get("qty", position.get("net_qty", "0")) or "0").strip()
    try:
        return Decimal(raw).copy_abs()
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _position_notional(position: dict) -> Decimal:
    raw = str(position.get("notional", "0") or "0").strip()
    try:
        return Decimal(raw).copy_abs()
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _position_side(position: dict) -> str:
    side = str(position.get("side") or position.get("position_side") or "").strip().lower()
    if side in ("buy", "long"):
        return "long"
    if side in ("sell", "short"):
        return "short"
    qty = str(position.get("net_qty", "") or "").strip()
    if qty.startswith("-"):
        return "short"
    return "long"


def _decimal_str(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _nonnegative_decimal(value: str, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a decimal value") from None
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _at_least_trade_notional(notional: Decimal, min_trade_notional: Decimal) -> Decimal:
    if min_trade_notional > 0:
        return max(notional, min_trade_notional)
    return notional


def _align_qty_down(qty: Decimal, qty_step: Decimal) -> Decimal:
    if qty_step <= 0 or qty <= 0:
        return qty
    units = (qty / qty_step).to_integral_value(rounding=ROUND_FLOOR)
    return units * qty_step


def _event_body(symbol: str, rule: str, trigger_price: str, oi_mc: str, z_score: str) -> dict:
    base = _normalize_symbol(symbol)
    coin = base[:-4] if base.endswith("USDC") else base
    fields = RULE_PRESETS[rule]
    return {
        "hl_symbol": coin,
        "binance_symbol": f"{coin}USDT",
        "trigger_price": str(trigger_price),
        "oi_mc": str(oi_mc),
        "z_score": str(z_score),
        **fields,
    }


def _debug_publish(client: TradingClient, bot_id: str, body: dict) -> dict:
    path = f"/agent/realtime/bots/{bot_id}/ambush-events/_debug_publish"
    return client._request("POST", path, body=body)


def _post_decision_best_effort(
    client: TradingClient,
    event_id: int | None,
    side: str,
    reason: str,
    source_order_id: str = "",
) -> None:
    if not event_id:
        return
    try:
        client.post_ambush_decision(
            int(event_id),
            "long" if side == "long" else "short",
            reason,
            momentum_passed=True,
            source_order_id=source_order_id,
        )
    except Exception as exc:  # pragma: no cover - best-effort QA logging
        LOG.warning("decision write-back failed event_id=%s: %s", event_id, exc)


def _open_position(client: TradingClient, side: str, notional: str, leverage: int, order_id: str) -> dict:
    zh = (
        f"跟单链路测试：skill 侧手动注入异动事件后执行{('做多' if side == 'long' else '做空')}。"
        "本轮用于验证 source-events、fills 和跟单消费链路，不代表真实策略判断。"
    )
    en = (
        f"Copy-trading test: the skill injected a synthetic ambush event and manually opened a "
        f"{side} position. This cycle validates source-events, fills, and follower consumption; "
        "it is not a live strategy judgement."
    )
    if side == "long":
        return client.open_long(notional, leverage, order_id, zh, en)
    return client.open_short(notional, leverage, order_id, zh, en)


def _reduce_position(client: TradingClient, position: dict, symbol: str, qty: Decimal, order_id: str) -> dict:
    side = str(position.get("side") or position.get("position_side") or "").upper()
    zh = "跟单链路测试：5分钟模拟决策为减仓，skill 侧只减部分仓位，保持 Agent 持续有仓。"
    en = (
        "Copy-trading test: the five-minute simulated decision is reduce. "
        "The skill only reduces part of the position so the agent remains invested."
    )
    return client.close_position(side, _decimal_str(qty), order_id, zh, en, symbol=symbol)


def _next_side(state_path: Path, mode: str, fixed_side: str) -> str:
    if fixed_side in ("long", "short"):
        return fixed_side
    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    last = state.get("last_open_side", "short")
    if mode == "alternate":
        return "short" if last == "long" else "long"
    return "long"


def _save_side(state_path: Path, side: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)
    state["last_open_side"] = side
    state["updated_at"] = int(time.time())
    state_path.write_text(json.dumps(state, indent=2))


def _load_state(state_path: Path) -> dict:
    if state_path.is_file():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return {}
    return {}


def _next_decision(state_path: Path, mode: str) -> str:
    if mode in ("add", "reduce"):
        return mode
    state = _load_state(state_path)
    last = state.get("last_decision", "reduce")
    return "reduce" if last == "add" else "add"


def _save_decision(state_path: Path, decision: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)
    state["last_decision"] = decision
    state["updated_at"] = int(time.time())
    state_path.write_text(json.dumps(state, indent=2))


def _inject_and_open(args: argparse.Namespace, client: TradingClient, creds: dict, side: str | None = None) -> None:
    symbol = _normalize_symbol(args.symbol)
    now = int(time.time())
    side = side or _next_side(Path(args.state_file).expanduser(), args.side_mode, args.side)
    rule = args.long_rule if side == "long" else args.short_rule
    event = _event_body(symbol, rule, args.trigger_price, args.oi_mc, args.z_score)
    LOG.info("injecting event side=%s rule=%s body=%s", side, rule, json.dumps(event, ensure_ascii=False))
    pub = _debug_publish(client, creds["bot_id"], event)
    LOG.info("debug_publish response: %s", json.dumps(pub, ensure_ascii=False))
    event_id = pub.get("event_id") if isinstance(pub, dict) else None

    order_id = f"ambush-followtest-open-{side}-{now}"
    min_trade_notional = _nonnegative_decimal(args.min_trade_notional, "--min-trade-notional")
    open_notional = _at_least_trade_notional(
        _nonnegative_decimal(args.notional, "--notional"),
        min_trade_notional,
    )
    resp = _open_position(client, side, _decimal_str(open_notional), args.leverage, order_id)
    LOG.info("open response: %s", json.dumps(resp, ensure_ascii=False))
    source_order_id = ""
    if isinstance(resp, dict):
        source_order_id = str(resp.get("order_id") or "")
    _post_decision_best_effort(
        client,
        event_id,
        side,
        "manual_followtest_open",
        source_order_id=source_order_id,
    )
    _save_side(Path(args.state_file).expanduser(), side)


def run_once(args: argparse.Namespace, client: TradingClient, creds: dict) -> None:
    symbol = _normalize_symbol(args.symbol)
    position = _position_for_symbol(client, symbol)
    now = int(time.time())
    min_notional = _nonnegative_decimal(args.min_notional, "--min-notional")
    min_trade_notional = _nonnegative_decimal(args.min_trade_notional, "--min-trade-notional")
    qty_step = _nonnegative_decimal(args.qty_step, "--qty-step")

    if position:
        current_notional = _position_notional(position)
        if args.always_position and current_notional < min_notional:
            side = _position_side(position)
            add_notional = _at_least_trade_notional(
                max(_nonnegative_decimal(args.add_notional, "--add-notional"), min_notional - current_notional),
                min_trade_notional,
            )
            order_id = f"ambush-followtest-topup-{side}-{now}"
            LOG.info(
                "notional below floor; adding symbol=%s side=%s current_notional=%s min_notional=%s add_notional=%s min_trade_notional=%s",
                symbol,
                side,
                current_notional,
                min_notional,
                add_notional,
                min_trade_notional,
            )
            resp = _open_position(client, side, _decimal_str(add_notional), args.leverage, order_id)
            LOG.info("topup response: %s", json.dumps(resp, ensure_ascii=False))
            _save_side(Path(args.state_file).expanduser(), side)
            _save_decision(Path(args.state_file).expanduser(), "add")
            return

        if not args.always_position:
            qty = _position_qty(position)
            order_id = f"ambush-followtest-close-{now}"
            LOG.info("closing existing %s position: %s", symbol, json.dumps(position, ensure_ascii=False))
            resp = _reduce_position(client, position, symbol, qty, order_id)
            LOG.info("close response: %s", json.dumps(resp, ensure_ascii=False))
            return

        state_path = Path(args.state_file).expanduser()
        decision = _next_decision(state_path, args.decision_mode)
        if decision == "add":
            side = _position_side(position)
            add_notional = _at_least_trade_notional(
                _nonnegative_decimal(args.add_notional, "--add-notional"),
                min_trade_notional,
            )
            order_id = f"ambush-followtest-add-{side}-{now}"
            LOG.info(
                "decision=add symbol=%s side=%s notional=%s min_trade_notional=%s",
                symbol,
                side,
                add_notional,
                min_trade_notional,
            )
            resp = _open_position(client, side, _decimal_str(add_notional), args.leverage, order_id)
            LOG.info("add response: %s", json.dumps(resp, ensure_ascii=False))
            _save_side(state_path, side)
            _save_decision(state_path, "add")
            return

        qty = _position_qty(position)
        reduce_ratio = max(Decimal("0.01"), min(Decimal(str(args.reduce_ratio)), Decimal("0.90")))
        reduce_notional = current_notional * reduce_ratio
        max_reduce_notional = max(Decimal("0"), current_notional - min_notional)
        reduce_notional = min(reduce_notional, max_reduce_notional)
        if reduce_notional > 0 and reduce_notional < min_trade_notional:
            reduce_notional = Decimal("0")
        reduce_qty = qty * (reduce_notional / current_notional) if current_notional > 0 else Decimal("0")
        reduce_qty = _align_qty_down(reduce_qty, qty_step)
        actual_reduce_notional = current_notional * (reduce_qty / qty) if qty > 0 else Decimal("0")
        if actual_reduce_notional > 0 and actual_reduce_notional < min_trade_notional:
            reduce_qty = Decimal("0")
            actual_reduce_notional = Decimal("0")
        if reduce_qty <= 0:
            side = _position_side(position)
            add_notional = _at_least_trade_notional(
                _nonnegative_decimal(args.add_notional, "--add-notional"),
                min_trade_notional,
            )
            LOG.info(
                "decision=reduce but notional floor blocks reduce; adding instead side=%s current_notional=%s min_notional=%s min_trade_notional=%s add_notional=%s",
                side,
                current_notional,
                min_notional,
                min_trade_notional,
                add_notional,
            )
            resp = _open_position(client, side, _decimal_str(add_notional), args.leverage, f"ambush-followtest-add-{side}-{now}")
            LOG.info("fallback add response: %s", json.dumps(resp, ensure_ascii=False))
            _save_decision(state_path, "add")
            return
        order_id = f"ambush-followtest-reduce-{now}"
        LOG.info(
            "decision=reduce symbol=%s notional=%s reduce_notional=%s actual_reduce_notional=%s qty=%s reduce_qty=%s min_notional=%s min_trade_notional=%s qty_step=%s",
            symbol,
            current_notional,
            reduce_notional,
            actual_reduce_notional,
            qty,
            reduce_qty,
            min_notional,
            min_trade_notional,
            qty_step,
        )
        resp = _reduce_position(client, position, symbol, reduce_qty, order_id)
        LOG.info("reduce response: %s", json.dumps(resp, ensure_ascii=False))
        _save_decision(state_path, "reduce")
        return

    _inject_and_open(args, client, creds)


def main() -> int:
    p = argparse.ArgumentParser(description="Run Ambush manual open/close cycles for copy-trading QA.")
    p.add_argument("--creds", required=True, help="Ambush bot creds JSON with api_key/api_secret/bot_id/base_url")
    p.add_argument("--platform-url", default="", help="Override creds.base_url")
    p.add_argument("--symbol", default="SAGAUSDC", help="Symbol to trade in the ambush bot")
    p.add_argument("--notional", default="100", help="Open notional per test trade")
    p.add_argument("--add-notional", default="50", help="Additional notional when the 5-minute decision is add")
    p.add_argument("--reduce-ratio", default="0.35", help="Fraction of current qty to reduce on reduce decisions")
    p.add_argument("--min-notional", default="0", help="Minimum open-position notional to maintain")
    p.add_argument("--min-trade-notional", default="0", help="Minimum notional for each add/open or reduce-only order; 0 disables")
    p.add_argument("--qty-step", default="0", help="Round reduce qty down to this step size before submit; 0 disables")
    p.add_argument("--leverage", type=int, default=3)
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--initial-delay-seconds", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=0, help="0 means run forever")
    p.add_argument(
        "--always-position", action="store_true",
        help="Keep the bot invested: existing positions receive add/reduce decisions instead of full close.",
    )
    p.add_argument("--decision-mode", choices=["alternate", "add", "reduce"], default="alternate")
    p.add_argument("--side", choices=["", "long", "short"], default="", help="Fixed side; empty uses side-mode")
    p.add_argument("--side-mode", choices=["long", "alternate"], default="alternate")
    p.add_argument("--long-rule", choices=sorted(RULE_PRESETS), default="long_momentum_init")
    p.add_argument("--short-rule", choices=sorted(RULE_PRESETS), default="short_spike_extreme")
    p.add_argument("--trigger-price", default="0.0208")
    p.add_argument("--oi-mc", default="0.45")
    p.add_argument("--z-score", default="3.0")
    p.add_argument("--state-file", default="~/.moss-trade-bot/ambush_followtest_sim_state.json")
    p.add_argument("--log-file", default="", help="Optional append log path")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    creds = _load_json(args.creds)
    base_url = args.platform_url or creds.get("base_url", "")
    if not base_url or not creds.get("bot_id"):
        raise SystemExit("creds must contain base_url and bot_id, or pass --platform-url")
    client = TradingClient(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        base_url=base_url,
        bot_id=creds["bot_id"],
        symbol=args.symbol,
    )

    cycle = 0
    if args.initial_delay_seconds > 0:
        LOG.info("initial delay %d seconds before first decision cycle", args.initial_delay_seconds)
        time.sleep(args.initial_delay_seconds)
    while True:
        cycle += 1
        LOG.info("cycle %d start bot_id=%s symbol=%s", cycle, creds["bot_id"], _normalize_symbol(args.symbol))
        try:
            run_once(args, client, creds)
        except Exception:
            LOG.exception("cycle %d failed", cycle)
        if args.max_cycles > 0 and cycle >= args.max_cycles:
            break
        time.sleep(max(1, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
