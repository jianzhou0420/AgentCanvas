#!/usr/bin/env bash
# =============================================================================
# scripts/install/lib/thirdparty.sh
# =============================================================================
# Single source of truth for the third_party/ repositories that used to be git
# submodules (removed 2026-06-30). Each per-env install script sources this file
# and calls `ensure_thirdparty <name>` to clone the repo into third_party/<name>
# and pin it to the EXACT commit recorded below.
#
# Why this exists:
#   - VLN-CE + habitat-lab are needed by BOTH install_ac_vlnce.sh and
#     install_ac_smartway.sh; libero by BOTH install_ac_vla_policy.sh and
#     install_ac_libero.sh. Keeping the pinned commit IDs in one place prevents
#     the two consumers from drifting apart.
#
#   - lerobot + libero are PUBLIC upstreams (formerly nested inside the private
#     jianzhou0420/vlaworkspace repo). They are cloned here directly from their
#     public origins so the install never touches the private repo. The small
#     vendored openpi-client lives in the policy_adapter_vla nodeset, not here.
#
# Idempotent: re-running clones if absent, otherwise fetches + re-checks-out the
# pinned commit. To bump a pin, edit the SHA here (one place) and re-run the
# install script.
#
# Usage:
#   source "$SCRIPT_DIR/lib/thirdparty.sh"
#   ensure_thirdparty VLN-CE
# =============================================================================

# Resolve PROJECT_ROOT relative to this lib file (.../scripts/install/lib/).
_TP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP_PROJECT_ROOT="$(cd "$_TP_LIB_DIR/../../.." && pwd)"
TP_DIR="$TP_PROJECT_ROOT/third_party"

# spec = "<git-url>|<pinned-sha>|<nested-submodule-init>"
#   nested field: empty = none; "--recursive" = all nested submodules;
#   otherwise a space-separated list of nested submodule paths to init.
_thirdparty_spec() {
    case "$1" in
        VLN-CE)
            echo "https://github.com/jianzhou0420/VLN-CE.git|26c05ba72247f38a8f36052c0799a727b5dc4218|" ;;
        habitat-lab)
            echo "https://github.com/facebookresearch/habitat-lab.git|d6ed1c0a0e786f16f261de2beafe347f4186d0d8|" ;;
        Matterport3DSimulator)
            echo "https://github.com/peteanderson80/Matterport3DSimulator.git|589d091b111333f9e9f9d6cfd021b2eb68435925|--recursive" ;;
        lerobot)
            echo "https://github.com/huggingface/lerobot.git|f6b16f6d97155e3ce34ab2a1ec145e9413588197|" ;;
        libero)
            echo "https://github.com/Lifelong-Robot-Learning/LIBERO.git|f78abd68ee283de9f9be3c8f7e2a9ad60246e95c|" ;;
        *)
            echo "" ;;
    esac
}

ensure_thirdparty() {
    local name="$1"
    local spec
    spec="$(_thirdparty_spec "$name")"
    if [ -z "$spec" ]; then
        echo "[thirdparty] ERROR: unknown repo '$name' (not registered in thirdparty.sh)" >&2
        return 1
    fi

    local url sha nested
    IFS='|' read -r url sha nested <<< "$spec"
    local dest="$TP_DIR/$name"

    # ── Clone if missing ──
    if [ ! -e "$dest/.git" ]; then
        echo "[thirdparty] cloning $name <- $url"
        git clone "$url" "$dest"
    fi

    # ── Pin to the recorded commit ──
    local cur
    cur="$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo none)"
    if [ "$cur" != "$sha" ]; then
        echo "[thirdparty] pinning $name to ${sha:0:10} (was ${cur:0:10})"
        git -C "$dest" fetch --tags origin "$sha" 2>/dev/null \
            || git -C "$dest" fetch --all --tags
        git -C "$dest" checkout --quiet "$sha"
    else
        echo "[thirdparty] $name already at pinned ${sha:0:10}"
    fi

    # ── Initialize nested submodules (their SHAs are pinned by the commit above) ──
    if [ -n "$nested" ]; then
        if [ "$nested" = "--recursive" ]; then
            echo "[thirdparty] init nested submodules (recursive) in $name"
            git -C "$dest" submodule update --init --recursive
        else
            echo "[thirdparty] init nested submodules in $name: $nested"
            git -C "$dest" submodule update --init $nested
        fi
    fi
}
