#!/usr/bin/env python3
"""Live trading CLI for the simulation platform.

Usage:
    # Bind (get credentials):
    python live_trade.py bind --platform-url https://ai.moss.site --pair-code ABCD-EFGH --name "利弗莫尔v2"
    # --platform-url should be site origin only, e.g. https://ai.moss.site

    # Check status:
    python live_trade.py status --key ak_xxx --secret as_xxx

    # Open long:
    python live_trade.py open-long --key ak_xxx --secret as_xxx --amount 1000 --leverage 10

    # Open short:
    python live_trade.py open-short --key ak_xxx --secret as_xxx --amount 1000 --leverage 10

    # Close position:
    python live_trade.py close --key ak_xxx --secret as_xxx --side LONG

    # Get price:
    python live_trade.py price --key ak_xxx --secret as_xxx

    # Order history:
    python live_trade.py orders --key ak_xxx --secret as_xxx

    # Trade history:
    python live_trade.py trades --key ak_xxx --secret as_xxx

    # Save/load credentials (use local persistent path, not /tmp):
    python live_trade.py bind ... --save ~/.moss-trade-bot/agent_creds.json
    python live_trade.py status --creds ~/.moss-trade-bot/agent_creds.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trading_client import TradingClient
from text_i18n import default_text, validate_bilingual_text

PLATFORM_URL_HELP = "Platform site origin only, e.g. https://ai.moss.site. The client appends API paths automatically."


def resolve_base_url(args, creds=None) -> str:
    creds = creds or {}
    return (
        getattr(args, "platform_url", "")
        or creds.get("base_url", "")
    )


def load_client(args) -> TradingClient:
    if hasattr(args, "creds") and args.creds:
        with open(args.creds) as f:
            creds = json.load(f)
        return TradingClient(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            base_url=resolve_base_url(args, creds),
            bot_id=creds.get("bot_id", ""),
        )
    return TradingClient(
        api_key=getattr(args, "key", ""),
        api_secret=getattr(args, "secret", ""),
        base_url=resolve_base_url(args),
        bot_id=getattr(args, "bot_id", ""),
    )


def cmd_bind(args):
    client = TradingClient(base_url=resolve_base_url(args))
    result = client.bind(
        args.pair_code,
        display_name=args.name,
        persona=getattr(args, "persona", "") or args.name,
        description=getattr(args, "description", "") or f"{args.name} trading bot",
        fingerprint=args.fingerprint or "",
    )
    print(json.dumps(result, indent=2))

    if "api_secret" in result and args.save:
        to_save = {
            "binding_id": result.get("binding_id", ""),
            "api_key": result["api_key"],
            "api_secret": result["api_secret"],
            "base_url": client.base_url,
        }
        with open(args.save, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"\nCredentials saved to {args.save} (bind only; create realtime bot with create-bot)", file=sys.stderr)


def cmd_create_bot(args):
    """Create a realtime bot under current binding; write bot_id into creds."""
    with open(args.creds) as f:
        creds = json.load(f)
    with open(args.params_file) as f:
        strategy_params = json.load(f)
    try:
        name_i18n = validate_bilingual_text(
            "display_name_i18n",
            {"zh": args.name_zh, "en": args.name_en},
            64,
        )
        persona_i18n = validate_bilingual_text(
            "persona_i18n",
            {"zh": args.persona_zh, "en": args.persona_en},
            64,
        )
        description_i18n = validate_bilingual_text(
            "description_i18n",
            {"zh": args.description_zh, "en": args.description_en},
            280,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    client = TradingClient(
        api_key=creds["api_key"],
        api_secret=creds["api_secret"],
        base_url=resolve_base_url(args, creds),
        bot_id=creds.get("bot_id", ""),
    )
    result = client.create_realtime_bot(
        display_name=args.name or default_text(name_i18n),
        persona=args.persona or default_text(persona_i18n),
        description=args.description or default_text(description_i18n),
        strategy_params=strategy_params,
        display_name_i18n=name_i18n,
        persona_i18n=persona_i18n,
        description_i18n=description_i18n,
    )
    print(json.dumps(result, indent=2))
    bot_id = result.get("bot_id") or result.get("id")
    if not bot_id:
        print("No bot_id in response; check API response shape.", file=sys.stderr)
        sys.exit(1)
    creds["bot_id"] = bot_id
    if client.base_url:
        creds["base_url"] = client.base_url
    with open(args.creds, "w") as f:
        json.dump(creds, f, indent=2)
    print(f"\nbot_id saved to {args.creds}", file=sys.stderr)


def cmd_status(args):
    client = load_client(args)
    account = client.get_account()
    positions = client.get_positions()
    price = client.get_price()

    print("=== Account ===")
    print(json.dumps(account, indent=2))
    print("\n=== Positions ===")
    print(json.dumps(positions, indent=2))
    print("\n=== Price ===")
    print(json.dumps(price, indent=2))


def cmd_price(args):
    client = load_client(args)
    print(json.dumps(client.get_price(), indent=2))


def cmd_open_long(args):
    client = load_client(args)
    result = client.open_long(args.amount, args.leverage, args.order_id or "")
    print(json.dumps(result, indent=2))


def cmd_open_short(args):
    client = load_client(args)
    result = client.open_short(args.amount, args.leverage, args.order_id or "")
    print(json.dumps(result, indent=2))


def cmd_close(args):
    client = load_client(args)
    result = client.close_position(args.side, args.qty or "")
    print(json.dumps(result, indent=2))


def cmd_orders(args):
    client = load_client(args)
    print(json.dumps(client.get_orders(args.limit), indent=2))


def cmd_trades(args):
    client = load_client(args)
    print(json.dumps(client.get_trades(args.limit), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Live trading CLI")
    sub = parser.add_subparsers(dest="command")

    # bind
    p = sub.add_parser("bind", help="Bind agent with pair code")
    p.add_argument("--pair-code", required=True)
    p.add_argument("--name", default="Bot")
    p.add_argument("--persona", default="", help="Bot persona (e.g. 趋势死磕派)")
    p.add_argument("--description", default="", help="Bot description")
    p.add_argument("--fingerprint", default="")
    p.add_argument("--save", default="", help="Save credentials to JSON file")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Saved into agent_creds.json for later reuse.")

    # create-bot
    p = sub.add_parser("create-bot", help="Create realtime bot (after bind); writes bot_id to creds")
    p.add_argument("--creds", required=True, help="Credentials JSON from bind")
    p.add_argument("--name", default="", help="Legacy projection only; prefer --name-zh/--name-en")
    p.add_argument("--name-zh", required=True, help="Bot display name (Chinese)")
    p.add_argument("--name-en", required=True, help="Bot display name (English)")
    p.add_argument("--persona", default="", help="Legacy projection only; prefer --persona-zh/--persona-en")
    p.add_argument("--persona-zh", required=True, help="Bot persona (Chinese)")
    p.add_argument("--persona-en", required=True, help="Bot persona (English)")
    p.add_argument("--description", default="", help="Legacy projection only; prefer --description-zh/--description-en")
    p.add_argument("--description-zh", required=True, help="Bot description (Chinese)")
    p.add_argument("--description-en", required=True, help="Bot description (English)")
    p.add_argument("--params-file", required=True, help="Strategy params JSON (e.g. params.json)")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # status
    p = sub.add_parser("status", help="Account + positions + price")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="", help="Load from credentials JSON")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # price
    p = sub.add_parser("price", help="Get mark price")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # open-long
    p = sub.add_parser("open-long", help="Open long position")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--amount", required=True, help="Notional USDT")
    p.add_argument("--leverage", type=int, required=True)
    p.add_argument("--order-id", default="")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # open-short
    p = sub.add_parser("open-short", help="Open short position")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--amount", required=True, help="Notional USDT")
    p.add_argument("--leverage", type=int, required=True)
    p.add_argument("--order-id", default="")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # close
    p = sub.add_parser("close", help="Close position")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--side", required=True, choices=["LONG", "SHORT"])
    p.add_argument("--qty", default="", help="BTC qty, empty=close all")
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # orders
    p = sub.add_parser("orders", help="Order history")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--platform-url", default="", help=PLATFORM_URL_HELP + " Otherwise reuse base_url from creds file.")

    # trades
    p = sub.add_parser("trades", help="Trade history")
    p.add_argument("--key", default="")
    p.add_argument("--secret", default="")
    p.add_argument("--creds", default="")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--platform-url", default="", help="Preferred explicit platform base URL; otherwise reuse base_url from creds file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "bind": cmd_bind, "create-bot": cmd_create_bot,
        "status": cmd_status, "price": cmd_price,
        "open-long": cmd_open_long, "open-short": cmd_open_short,
        "close": cmd_close, "orders": cmd_orders, "trades": cmd_trades,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
