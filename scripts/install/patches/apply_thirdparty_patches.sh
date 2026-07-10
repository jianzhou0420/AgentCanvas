#!/bin/bash
# =============================================================================
# Apply third-party patches
# =============================================================================
# Re-applies our local patches to the third_party/ sources (cloned + pinned by
# the per-env install scripts via scripts/install/lib/thirdparty.sh).
# Run this once after those sources are fetched, and again any time a source is
# re-checked-out to its pinned commit (which discards working-tree edits).
#
# Idempotent: each patch is checked with `git apply --reverse --check` first;
# if it's already applied the step is skipped. Exit code is 0 on success.
#
# Patches applied:
#   1. VLN-CE/setup.py                       — makes vlnce / habitat_extensions
#                                              / vlnce_server pip-installable.
#   2. libero/benchmark/__init__.py          — PyTorch 2.6+ weights_only=False
#                                              fallback for init_states pickles.
#   3. libero/envs/venv.py                   — gym -> gymnasium import (libero
#                                              upstream hasn't migrated).
#
# Usage:
#   bash scripts/install/patches/apply_thirdparty_patches.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PATCH_DIR="$SCRIPT_DIR"

echo "=== Applying third-party submodule patches ==="
echo "Project root: $PROJECT_ROOT"
echo "Patch dir:    $PATCH_DIR"
echo ""

# -----------------------------------------------------------------------------
# Helper: apply a .patch file inside a submodule, idempotent.
#   $1 = submodule path relative to PROJECT_ROOT
#   $2 = patch file (absolute path)
# -----------------------------------------------------------------------------
apply_patch() {
    local submodule="$1"
    local patch_file="$2"
    local sm_dir="$PROJECT_ROOT/$submodule"
    local patch_name
    patch_name="$(basename "$patch_file")"

    if [ ! -d "$sm_dir/.git" ] && [ ! -f "$sm_dir/.git" ]; then
        echo "  [skip] $submodule: source not fetched (run the env's install_ac_*.sh first)"
        return 0
    fi
    if [ ! -f "$patch_file" ]; then
        echo "  [fail] patch file missing: $patch_file" >&2
        return 1
    fi

    pushd "$sm_dir" > /dev/null

    if git apply --reverse --check "$patch_file" > /dev/null 2>&1; then
        echo "  [ok]   $submodule :: $patch_name (already applied)"
    elif git apply --check "$patch_file" > /dev/null 2>&1; then
        git apply "$patch_file"
        echo "  [done] $submodule :: $patch_name (applied)"
    else
        echo "  [fail] $submodule :: $patch_name (cannot apply cleanly)" >&2
        echo "         submodule HEAD may have moved; regenerate the patch." >&2
        popd > /dev/null
        return 1
    fi

    popd > /dev/null
}

# -----------------------------------------------------------------------------
# Helper: copy a static file into a submodule if missing or different.
#   $1 = submodule path relative to PROJECT_ROOT
#   $2 = source file (absolute path)
#   $3 = destination path relative to submodule
# -----------------------------------------------------------------------------
install_file() {
    local submodule="$1"
    local src="$2"
    local dst_rel="$3"
    local dst="$PROJECT_ROOT/$submodule/$dst_rel"

    if [ ! -d "$PROJECT_ROOT/$submodule" ]; then
        echo "  [skip] $submodule: directory missing (run the env's install_ac_*.sh first)"
        return 0
    fi
    if [ ! -f "$src" ]; then
        echo "  [fail] source missing: $src" >&2
        return 1
    fi

    if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        echo "  [ok]   $submodule :: $dst_rel (already installed)"
    else
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "  [done] $submodule :: $dst_rel (installed)"
    fi
}

# -----------------------------------------------------------------------------
# Apply patches
# -----------------------------------------------------------------------------
echo "[1/3] VLN-CE setup.py"
install_file \
    "third_party/VLN-CE" \
    "$PATCH_DIR/VLN-CE_setup.py" \
    "setup.py"

echo ""
echo "[2/3] libero benchmark torch.load weights_only fallback"
apply_patch \
    "third_party/libero" \
    "$PATCH_DIR/libero_benchmark_init.patch"

echo ""
echo "[3/3] libero venv gym -> gymnasium"
apply_patch \
    "third_party/libero" \
    "$PATCH_DIR/libero_venv_gymnasium.patch"

echo ""
echo "[extra] libero package-root __init__.py"
# LIBERO upstream forgets to ship libero/__init__.py — without it,
# find_packages() in its setup.py finds nothing and `pip install -e` registers
# no package, so `import libero` fails. Centralized here (idempotent) so BOTH
# install_ac_libero.sh and install_ac_vla_policy.sh get it via this applier.
LIBERO_INIT="$PROJECT_ROOT/third_party/libero/libero/__init__.py"
if [ ! -d "$PROJECT_ROOT/third_party/libero" ]; then
    echo "  [skip] third_party/libero not fetched"
elif [ -f "$LIBERO_INIT" ]; then
    echo "  [ok]   third_party/libero/libero/__init__.py (present)"
else
    touch "$LIBERO_INIT"
    echo "  [done] third_party/libero/libero/__init__.py (touched)"
fi

echo ""
echo "=== Done ==="
