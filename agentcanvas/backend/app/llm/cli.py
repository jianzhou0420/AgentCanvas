"""Standalone CLI for AgentCanvas profile management.

Entry points:
    python -m agentcanvas.backend.app.cli config <command>
    python -m agentcanvas.backend.app config <command>

Does NOT require the FastAPI server to be running.
Changes are picked up by the running server within ~2 seconds (mtime invalidation).

API keys are read from each provider's standard env var (e.g. ``OPENAI_API_KEY``
for ``openai``). Profiles never store keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .profiles import LLMProfile, get_profile_store
from .providers import PROVIDER_REGISTRY, resolve_provider_config

# ── Key masking ────────────────────────────────────────────────────────────────


def _mask_key(key: str) -> str:
    """Mask an API key: show first 3 + last 4 chars, or '****' if too short."""
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "****"
    return f"{key[:3]}...{key[-4:]}"


# ── Subcommand implementations ─────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> int:
    store = get_profile_store()
    profiles = store.list_profiles()
    active = store.get_active()

    if args.json:
        out = []
        for name, p in profiles.items():
            cfg = resolve_provider_config(p)
            out.append(
                {
                    "name": name,
                    "provider": p.provider,
                    "model": cfg["model"],
                    "api_type": cfg["api_type"],
                    "has_key": bool(cfg["api_key"]) or cfg["api_type"] == "ollama",
                    "active": name == active,
                }
            )
        print(json.dumps(out, indent=2))
        return 0

    if not profiles:
        print("No profiles configured.")
        print(
            "  Run: python -m agentcanvas.backend.app config set <name> --provider <id> --model <m>"
        )
        return 0

    col_name = max(len("NAME"), max(len(n) for n in profiles)) + 2
    col_prov = max(len("PROVIDER"), max(len(p.provider) for p in profiles.values())) + 2
    col_model = (
        max(len("MODEL"), max(len(resolve_provider_config(p)["model"]) for p in profiles.values()))
        + 2
    )

    header = (
        f"{'NAME':<{col_name}}{'PROVIDER':<{col_prov}}{'MODEL':<{col_model}}"
        f"{'HAS_KEY':<9}{'ACTIVE'}"
    )
    print(header)
    print("-" * len(header))

    for name, p in profiles.items():
        cfg = resolve_provider_config(p)
        is_active = name == active
        star = "*" if is_active else ""
        has_key = "yes" if cfg["api_key"] else ("n/a" if cfg["api_type"] == "ollama" else "no")
        print(
            f"{name:<{col_name}}{p.provider:<{col_prov}}{cfg['model']:<{col_model}}"
            f"{has_key:<9}{star}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = get_profile_store()
    profile = store.get(args.name)
    if profile is None:
        print(f"Error: profile '{args.name}' not found.", file=sys.stderr)
        return 1

    cfg = resolve_provider_config(profile)
    active = store.get_active()
    reg = PROVIDER_REGISTRY.get(profile.provider)
    env_var = reg.env_var if reg else ""
    api_key = cfg["api_key"]
    key_display = api_key if args.show_keys else _mask_key(api_key)

    if args.json:
        out = {
            "name": args.name,
            "provider": profile.provider,
            "model": cfg["model"],
            "api_type": cfg["api_type"],
            "base_url": cfg["base_url"],
            "env_var": env_var,
            "api_key": key_display,
            "active": args.name == active,
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"Profile: {args.name}{'  (active)' if args.name == active else ''}")
    print(f"  provider : {profile.provider}")
    print(f"  model    : {cfg['model']}")
    print(f"  api_type : {cfg['api_type']}")
    print(f"  base_url : {cfg['base_url'] or '(registry default)'}")
    print(f"  env_var  : {env_var or '(none — local server)'}")
    print(f"  api_key  : {key_display}")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    store = get_profile_store()
    existing = store.get(args.name)

    if existing is None:
        provider = args.provider or "custom"
        model = args.model or ""
        profile = LLMProfile(
            provider=provider,
            model=model,
            base_url=args.base_url or "",
            api_type=args.api_type or "",
        )
        try:
            store.create(args.name, profile)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps({"ok": True, "action": "created", "name": args.name}))
        else:
            print(f"Profile '{args.name}' created.")
            reg = PROVIDER_REGISTRY.get(profile.provider)
            if reg and reg.env_var:
                print(f"  Set the API key via env: export {reg.env_var}=<your-key>")
    else:
        fields: dict = {}
        if args.provider is not None:
            fields["provider"] = args.provider
        if args.model is not None:
            fields["model"] = args.model
        if args.base_url is not None:
            fields["base_url"] = args.base_url
        if args.api_type is not None:
            fields["api_type"] = args.api_type

        try:
            store.update(args.name, **fields)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "action": "updated",
                        "name": args.name,
                        "fields": list(fields.keys()),
                    }
                )
            )
        else:
            print(f"Profile '{args.name}' updated.")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    store = get_profile_store()
    try:
        store.delete(args.name)
    except KeyError:
        print(f"Error: profile '{args.name}' not found.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, "deleted": args.name}))
    else:
        print(f"Profile '{args.name}' deleted.")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    store = get_profile_store()
    name = args.name
    try:
        store.set_active(name)
    except KeyError:
        print(f"Error: profile '{name}' not found.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"ok": True, "active": name}))
    else:
        if name:
            print(f"Active profile set to '{name}'.")
        else:
            print("Active profile cleared.")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    from .key_validator import validate_api_key_sync

    store = get_profile_store()
    name = args.name or store.get_active()

    if not name:
        print("Error: no profile name given and no active profile set.", file=sys.stderr)
        print("  Usage: config test <name>  or  config activate <name> first", file=sys.stderr)
        return 1

    profile = store.get(name)
    if profile is None:
        print(f"Error: profile '{name}' not found.", file=sys.stderr)
        return 1

    cfg = resolve_provider_config(profile)
    api_key = cfg["api_key"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    api_type = cfg["api_type"]

    if not args.json:
        print(f"Testing profile '{name}' ({api_type}, {model}) ...", end=" ", flush=True)

    litellm_prefix = cfg.get("litellm_prefix", "")
    ok, msg = validate_api_key_sync(
        api_key, base_url, model, api_type, timeout=15.0, litellm_prefix=litellm_prefix
    )

    if args.json:
        print(json.dumps({"name": name, "ok": ok, "message": msg}))
    else:
        if ok:
            print(f"Success! ({msg})")
        else:
            print(f"Failed: {msg}")

    return 0 if ok else 1


def cmd_providers(args: argparse.Namespace) -> int:
    if args.json:
        out = {
            pid: {
                "label": p.label,
                "base_url": p.base_url,
                "api_type": p.api_type,
                "default_model": p.default_model,
                "env_var": p.env_var,
            }
            for pid, p in PROVIDER_REGISTRY.items()
        }
        print(json.dumps(out, indent=2))
        return 0

    col_id = max(len("ID"), max(len(k) for k in PROVIDER_REGISTRY)) + 2
    col_label = max(len("LABEL"), max(len(p.label) for p in PROVIDER_REGISTRY.values())) + 2
    col_type = max(len("API_TYPE"), max(len(p.api_type) for p in PROVIDER_REGISTRY.values())) + 2
    col_env = (
        max(len("ENV_VAR"), max(len(p.env_var) for p in PROVIDER_REGISTRY.values() if p.env_var))
        + 2
    )

    header = f"{'ID':<{col_id}}{'LABEL':<{col_label}}{'API_TYPE':<{col_type}}{'ENV_VAR':<{col_env}}"
    print(header)
    print("-" * len(header))
    for pid, p in PROVIDER_REGISTRY.items():
        env_display = p.env_var or "-"
        print(
            f"{pid:<{col_id}}{p.label:<{col_label}}{p.api_type:<{col_type}}{env_display:<{col_env}}"
        )
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    """Show standard env vars for each provider and the active profile."""
    show_keys = getattr(args, "show_keys", False)
    use_json = getattr(args, "json", False)

    rows = []
    for pid, p in PROVIDER_REGISTRY.items():
        if not p.env_var:
            continue
        val = os.environ.get(p.env_var, "")
        rows.append(
            {
                "provider": pid,
                "env_var": p.env_var,
                "set": bool(val),
                "value": val if show_keys else (_mask_key(val) if val else "(not set)"),
            }
        )

    store = get_profile_store()
    active = store.get_active()
    active_profile = store.get(active) if active else None

    if use_json:
        out: dict = {"env_vars": rows, "active_profile": active or None}
        if active_profile:
            cfg = resolve_provider_config(active_profile)
            out["active_resolved"] = {
                "provider": active_profile.provider,
                "model": cfg["model"],
                "api_type": cfg["api_type"],
                "key_set": bool(cfg["api_key"]),
            }
        print(json.dumps(out, indent=2))
        return 0

    print("Provider env vars:")
    for r in rows:
        marker = "✓" if r["set"] else "·"
        print(f"  {marker} {r['env_var']:<24} {r['value']}")

    print()
    print(f"Active profile: {active or '(none)'}")
    if active_profile:
        cfg = resolve_provider_config(active_profile)
        print(f"  provider : {active_profile.provider}")
        print(f"  model    : {cfg['model']}")
        print(f"  api_type : {cfg['api_type']}")
        print(f"  key_set  : {bool(cfg['api_key'])}")
        if not cfg["api_key"] and cfg["api_type"] != "ollama":
            reg = PROVIDER_REGISTRY.get(active_profile.provider)
            if reg and reg.env_var:
                print(f"  → set:    export {reg.env_var}=<your-key>")
    return 0


# ── Argument parser ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcanvas config",
        description="AgentCanvas LLM profile management CLI",
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--json", action="store_true", help="Output JSON instead of human-readable text"
    )
    shared.add_argument(
        "--show-keys",
        action="store_true",
        dest="show_keys",
        help="Show full API keys (default: masked)",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sub.add_parser("list", parents=[shared], help="List all profiles")

    p_show = sub.add_parser("show", parents=[shared], help="Show a single profile")
    p_show.add_argument("name", help="Profile name")

    p_set = sub.add_parser("set", parents=[shared], help="Create or update a profile (upsert)")
    p_set.add_argument("name", help="Profile name")
    p_set.add_argument("--provider", help="Provider ID (from 'config providers')")
    p_set.add_argument("--model", help="Model name")
    p_set.add_argument("--base-url", dest="base_url", help="Custom base URL")
    p_set.add_argument(
        "--api-type", dest="api_type", help="API type override (openai/anthropic/google/ollama)"
    )

    p_del = sub.add_parser("delete", parents=[shared], help="Delete a profile")
    p_del.add_argument("name", help="Profile name")

    p_act = sub.add_parser("activate", parents=[shared], help="Set the active profile")
    p_act.add_argument("name", nargs="?", default="", help="Profile name (empty to clear active)")

    p_test = sub.add_parser("test", parents=[shared], help="Test API key connectivity")
    p_test.add_argument(
        "name", nargs="?", default="", help="Profile name (default: active profile)"
    )

    sub.add_parser("providers", parents=[shared], help="List available providers from the registry")

    sub.add_parser("env", parents=[shared], help="Show provider env vars and active profile")

    return parser


# ── Dispatch ───────────────────────────────────────────────────────────────────


_DISPATCH = {
    "list": cmd_list,
    "show": cmd_show,
    "set": cmd_set,
    "delete": cmd_delete,
    "activate": cmd_activate,
    "test": cmd_test,
    "providers": cmd_providers,
    "env": cmd_env,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
