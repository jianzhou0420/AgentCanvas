#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AgentCanvas LLM Setup & Verification
#
# Interactive script to configure LLM providers and verify the config system.
#
# Usage:
#   bash agentcanvas/setup_llm.sh              # Interactive setup
#   bash agentcanvas/setup_llm.sh --verify     # Verify existing config only
#   bash agentcanvas/setup_llm.sh --list       # List profiles
#   bash agentcanvas/setup_llm.sh --reset      # Delete all profiles
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-${HOME}/miniforge3/envs/ac-vlnce/bin/python}"
CLI="$PY -m agentcanvas.backend.app.cli"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

cd "$ROOT"

# ── Preflight ─────────────────────────────────────────────────────────────────

preflight() {
    echo -e "${BOLD}Preflight checks${NC}"

    if [ ! -x "$PY" ]; then
        fail "Python not found at $PY"
        echo "  Set PYTHON=/path/to/python and re-run."
        exit 1
    fi
    ok "Python: $($PY --version 2>&1)"

    if ! $PY -c "from agentcanvas.backend.app.cli import main" 2>/dev/null; then
        fail "CLI module not importable — is the vlnce env set up?"
        exit 1
    fi
    ok "CLI module importable"

    if ! $PY -c "from agentcanvas.backend.app.key_validator import validate_api_key_sync" 2>/dev/null; then
        fail "key_validator module not importable"
        exit 1
    fi
    ok "Key validator importable"

    echo ""
}

# ── Verify ────────────────────────────────────────────────────────────────────

verify() {
    echo -e "${BOLD}Verifying config system${NC}"
    local pass=0
    local total=0

    # Test 1: Provider registry
    total=$((total + 1))
    if $PY -c "
from agentcanvas.backend.app.providers import PROVIDER_REGISTRY
assert PROVIDER_REGISTRY['openai'].default_model == 'gpt-4o'
" 2>/dev/null; then
        ok "Provider registry: openai default = gpt-4o"
        pass=$((pass + 1))
    else
        fail "Provider registry: bad default model"
    fi

    # Test 2: Config singleton
    total=$((total + 1))
    if $PY -c "
from agentcanvas.backend.app.config import get_settings
assert get_settings() is get_settings()
" 2>/dev/null; then
        ok "Config singleton: identity check"
        pass=$((pass + 1))
    else
        fail "Config singleton: broken"
    fi

    # Test 3: Env var fallback (uses explicit empty profile to isolate from active profile)
    total=$((total + 1))
    if AGENTCANVAS_API_KEY=sk-verify-test AGENTCANVAS_MODEL=gpt-4o $PY -c "
from agentcanvas.backend.app.llm import get_llm_config
# Pass a non-existent profile name so active profile doesn't interfere,
# then fall through to env-only config
cfg = get_llm_config('__nonexistent__')
if cfg is None:
    # Fallback: test env-only path directly
    from agentcanvas.backend.app.llm import _env_only_config
    cfg = _env_only_config()
assert cfg is not None and cfg.api_key == 'sk-verify-test' and cfg.model == 'gpt-4o'
" 2>/dev/null; then
        ok "Env var fallback: AGENTCANVAS_API_KEY works"
        pass=$((pass + 1))
    else
        fail "Env var fallback: broken"
    fi

    # Test 4: Legacy env var alias
    total=$((total + 1))
    if VLM_API_KEY=sk-legacy-test VLM_MODEL=gpt-4o $PY -c "
from agentcanvas.backend.app.llm import _env_only_config
cfg = _env_only_config()
assert cfg is not None and cfg.api_key == 'sk-legacy-test'
" 2>/dev/null; then
        ok "Legacy env vars: VLM_API_KEY alias works"
        pass=$((pass + 1))
    else
        fail "Legacy env vars: broken"
    fi

    # Test 5: CLI commands
    total=$((total + 1))
    if $CLI config providers --json 2>/dev/null | $PY -c "import json,sys; d=json.load(sys.stdin); assert 'openai' in d" 2>/dev/null; then
        ok "CLI: 'config providers --json' works"
        pass=$((pass + 1))
    else
        fail "CLI: 'config providers' broken"
    fi

    # Test 6: Profile round-trip
    total=$((total + 1))
    if $CLI config set __verify_tmp__ --provider openai --model gpt-4o --api-key sk-roundtrip 2>/dev/null && \
       $CLI config show __verify_tmp__ --json 2>/dev/null | $PY -c "import json,sys; d=json.load(sys.stdin); assert d['model']=='gpt-4o'" 2>/dev/null && \
       $CLI config delete __verify_tmp__ 2>/dev/null; then
        ok "Profile round-trip: create → show → delete"
        pass=$((pass + 1))
    else
        fail "Profile round-trip: broken"
        $CLI config delete __verify_tmp__ 2>/dev/null || true
    fi

    # Test 7: ProfileStore mtime (schema_version)
    total=$((total + 1))
    if $CLI config set __mtime_tmp__ --provider openai --model gpt-4o --api-key sk-mtime 2>/dev/null && \
       $PY -c "
import json
from pathlib import Path
p = Path.home() / '.agentcanvas' / 'profiles.json'
d = json.loads(p.read_text())
assert d.get('schema_version') == 1, f'Got {d.get(\"schema_version\")}'
" 2>/dev/null && \
       $CLI config delete __mtime_tmp__ 2>/dev/null; then
        ok "ProfileStore: schema_version = 1 in profiles.json"
        pass=$((pass + 1))
    else
        fail "ProfileStore: schema_version missing"
        $CLI config delete __mtime_tmp__ 2>/dev/null || true
    fi

    # Test 8: __main__.py entry point
    total=$((total + 1))
    if $PY -m agentcanvas.backend.app config providers --json 2>/dev/null | $PY -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
        ok "Entry point: python -m agentcanvas.backend.app config works"
        pass=$((pass + 1))
    else
        fail "Entry point: __main__.py broken"
    fi

    echo ""
    if [ "$pass" -eq "$total" ]; then
        echo -e "  ${GREEN}${BOLD}All $total checks passed!${NC}"
    else
        echo -e "  ${RED}${BOLD}$pass/$total checks passed${NC}"
    fi
    echo ""
    return $(( total - pass ))
}

# ── Interactive Setup ─────────────────────────────────────────────────────────

pick_provider() {
    echo -e "${BOLD}Select a provider${NC}"
    echo ""

    local providers=(openai anthropic google deepseek ollama openrouter together mistral xai)
    local labels=("OpenAI" "Anthropic" "Google Gemini" "DeepSeek" "Ollama (local)" "OpenRouter" "Together AI" "Mistral AI" "xAI (Grok)")

    for i in "${!providers[@]}"; do
        local n=$((i + 1))
        printf "  ${CYAN}%d${NC}) %-14s %s\n" "$n" "${providers[$i]}" "${labels[$i]}"
    done
    printf "  ${CYAN}0${NC}) %-14s %s\n" "custom" "Enter custom provider details"
    echo ""

    while true; do
        read -rp "  Choice [1-${#providers[@]}, 0 for custom]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 0 ] && [ "$choice" -le "${#providers[@]}" ]; then
            if [ "$choice" -eq 0 ]; then
                SELECTED_PROVIDER="custom"
            else
                SELECTED_PROVIDER="${providers[$((choice - 1))]}"
            fi
            break
        fi
        warn "Invalid choice"
    done
    echo ""
}

setup_interactive() {
    echo -e "\n${BOLD}${BLUE}━━━ AgentCanvas LLM Setup ━━━${NC}\n"

    # Show existing profiles
    local existing
    existing=$($CLI config list --json 2>/dev/null)
    local count
    count=$(echo "$existing" | $PY -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

    if [ "$count" -gt "0" ]; then
        echo -e "${BOLD}Current profiles:${NC}"
        $CLI config list 2>/dev/null
        echo ""
        read -rp "  Add another profile? [Y/n] " yn
        case "${yn,,}" in
            n|no) echo ""; return 0 ;;
        esac
        echo ""
    fi

    # Pick provider
    pick_provider

    # Profile name
    local default_name="$SELECTED_PROVIDER"
    read -rp "  Profile name [$default_name]: " profile_name
    profile_name="${profile_name:-$default_name}"

    # Model
    local default_model
    default_model=$($PY -c "
from agentcanvas.backend.app.providers import PROVIDER_REGISTRY
p = PROVIDER_REGISTRY.get('$SELECTED_PROVIDER')
print(p.default_model if p else '')
" 2>/dev/null || echo "")

    if [ -n "$default_model" ]; then
        read -rp "  Model [$default_model]: " model
        model="${model:-$default_model}"
    else
        read -rp "  Model: " model
    fi

    # API key (skip for ollama)
    local api_key=""
    if [ "$SELECTED_PROVIDER" = "ollama" ]; then
        info "Ollama runs locally — no API key needed"
    else
        echo -e "  ${BOLD}API Key${NC} (input hidden):"
        read -rsp "  > " api_key
        echo ""
        if [ -z "$api_key" ]; then
            warn "No API key provided — you can add it later with:"
            echo "    $CLI config set $profile_name --api-key"
        fi
    fi

    # Custom provider extras
    local extra_args=""
    if [ "$SELECTED_PROVIDER" = "custom" ]; then
        read -rp "  Base URL: " base_url
        read -rp "  API type [openai]: " api_type
        api_type="${api_type:-openai}"
        extra_args="--base-url $base_url --api-type $api_type"
    fi

    # Create profile
    echo ""
    local cmd_args="config set $profile_name --provider $SELECTED_PROVIDER --model $model"
    if [ -n "$api_key" ]; then
        echo "$api_key" | $CLI $cmd_args --api-key - $extra_args 2>/dev/null
    else
        $CLI $cmd_args $extra_args 2>/dev/null
    fi

    # Activate
    read -rp "  Set '$profile_name' as active profile? [Y/n] " yn
    case "${yn,,}" in
        n|no) ;;
        *)
            $CLI config activate "$profile_name" 2>/dev/null
            ;;
    esac

    # Test connectivity
    if [ -n "$api_key" ]; then
        echo ""
        read -rp "  Test API key now? [Y/n] " yn
        case "${yn,,}" in
            n|no) ;;
            *)
                echo -n "  Testing..."
                local result
                result=$($CLI config test "$profile_name" --json 2>/dev/null || echo '{"ok":false,"message":"CLI error"}')
                local test_ok
                test_ok=$(echo "$result" | $PY -c "import json,sys; print(json.load(sys.stdin).get('ok',False))" 2>/dev/null || echo "False")
                local test_msg
                test_msg=$(echo "$result" | $PY -c "import json,sys; print(json.load(sys.stdin).get('message','unknown'))" 2>/dev/null || echo "unknown")

                if [ "$test_ok" = "True" ]; then
                    echo -e " ${GREEN}✓ Connected!${NC} ($test_msg)"
                else
                    echo -e " ${RED}✗ Failed:${NC} $test_msg"
                    warn "You can re-test later: $CLI config test $profile_name"
                fi
                ;;
        esac
    fi

    # Summary
    echo ""
    echo -e "${BOLD}Summary:${NC}"
    $CLI config show "$profile_name" 2>/dev/null
    echo ""
    echo -e "${GREEN}Done!${NC} The server will pick up this profile automatically."
    echo ""
}

# ── Reset ─────────────────────────────────────────────────────────────────────

reset_profiles() {
    echo -e "${BOLD}${RED}Reset all profiles${NC}"
    echo ""
    $CLI config list 2>/dev/null
    echo ""
    read -rp "  Delete ALL profiles? This cannot be undone. [y/N] " yn
    case "${yn,,}" in
        y|yes)
            local names
            names=$($CLI config list --json 2>/dev/null | $PY -c "
import json, sys
for p in json.load(sys.stdin):
    print(p['name'])
" 2>/dev/null)
            while IFS= read -r name; do
                [ -z "$name" ] && continue
                $CLI config delete "$name" 2>/dev/null
                ok "Deleted: $name"
            done <<< "$names"
            echo ""
            ok "All profiles deleted."
            ;;
        *)
            info "Cancelled."
            ;;
    esac
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    case "${1:-}" in
        --verify|-v)
            preflight
            verify
            ;;
        --list|-l)
            $CLI config list 2>/dev/null
            ;;
        --reset)
            preflight
            reset_profiles
            ;;
        --env)
            $CLI config env 2>/dev/null
            ;;
        --help|-h)
            echo "Usage: bash agentcanvas/setup_llm.sh [OPTION]"
            echo ""
            echo "Options:"
            echo "  (none)      Interactive LLM provider setup"
            echo "  --verify    Run config system verification (8 checks)"
            echo "  --list      List configured profiles"
            echo "  --env       Show env var resolution"
            echo "  --reset     Delete all profiles"
            echo "  --help      Show this help"
            echo ""
            echo "After setup, start the server:"
            echo "  cd agentcanvas && bash run_dev.sh"
            echo ""
            echo "CLI reference:"
            echo "  python -m agentcanvas.backend.app config list"
            echo "  python -m agentcanvas.backend.app config set <name> --provider <id> --api-key <key>"
            echo "  python -m agentcanvas.backend.app config activate <name>"
            echo "  python -m agentcanvas.backend.app config test [<name>]"
            echo "  python -m agentcanvas.backend.app config providers"
            echo "  python -m agentcanvas.backend.app config env"
            ;;
        *)
            preflight
            setup_interactive
            echo -e "${BOLD}Verify installation:${NC}"
            verify
            ;;
    esac
}

main "$@"
