#!/bin/bash
# =============================================================================
# RT-1-X Checkpoint Download Script
# =============================================================================
# Downloads the RT-1-X TF SavedModel (~600MB zip) from the Open X-Embodiment
# GCS bucket into data/vla_policy/checkpoints/rt_1_x/.
#
# This script is checkpoint-download ONLY. The Python deps (tensorflow,
# tf-agents, tensorflow-hub, transforms3d, ...) are installed by
# scripts/install/install_ac_vla_policy.sh into the shared ac-vla-policy
# conda env — RT-1-X is part of policy_adapter_vla, not a separate env.
#
# Usage:
#   bash scripts/data/fetch_ckpt_rt1.sh
#
# Prerequisites:
#   - gsutil (Google Cloud SDK):
#     https://cloud.google.com/storage/docs/gsutil_install
#   - unzip
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT_DIR="$PROJECT_ROOT/data/vla_policy/checkpoints/rt_1_x"
CKPT_NAME="rt_1_x_tf_trained_for_002272480_step"
GS_URL="gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/${CKPT_NAME}.zip"

echo "=== RT-1-X Checkpoint Download ==="
echo "Project root:    $PROJECT_ROOT"
echo "Checkpoint dir:  $CKPT_DIR"
echo "Source:          $GS_URL"
echo ""

if [ -d "$CKPT_DIR/$CKPT_NAME" ] && [ -f "$CKPT_DIR/$CKPT_NAME/saved_model.pb" ]; then
    echo "  exists: $CKPT_DIR/$CKPT_NAME — skipping download"
    echo ""
    echo "=== Done ==="
    exit 0
fi

if ! command -v gsutil &> /dev/null; then
    echo "[ERROR] gsutil not found. Install Google Cloud SDK first:"
    echo "        https://cloud.google.com/storage/docs/gsutil_install"
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "[ERROR] unzip not found. Install with: sudo apt-get install unzip"
    exit 1
fi

mkdir -p "$CKPT_DIR"
TMP_ZIP="$CKPT_DIR/${CKPT_NAME}.zip"
echo "  fetching: $GS_URL"
echo "  →         $TMP_ZIP"
gsutil -m cp -r "$GS_URL" "$CKPT_DIR/"
echo "  unzipping into $CKPT_DIR/"
( cd "$CKPT_DIR" && unzip -q "$TMP_ZIP" && rm "$TMP_ZIP" )

if [ ! -f "$CKPT_DIR/$CKPT_NAME/saved_model.pb" ]; then
    echo "[ERROR] saved_model.pb not found after unzip — verify $CKPT_DIR contents"
    exit 1
fi

echo "  OK: $CKPT_DIR/$CKPT_NAME/"
echo ""
echo "=== Done ==="
echo ""
echo "RT-1-X is exposed through policy_adapter_vla as a (Rt1Model, Rt1Policy) pair."
echo "  Adapter:   workspace/nodesets/policy/policy_adapter_vla/adapters/models/rt1_model.py"
echo "  Policy:    workspace/nodesets/policy/policy_adapter_vla/policies/rt1_policy.py"
echo "  Inference: TF SavedModel wrapper embedded in policies/rt1_policy.py"
echo ""
echo "Next:"
echo "  1. Make sure scripts/install/install_ac_vla_policy.sh has been run"
echo "     (it installs the TF stack into ac-vla-policy)."
echo "  2. cd agentcanvas && bash run_dev.sh"
echo "  3. POST /api/components/nodesets/policy_adapter_vla/load?mode=server"
echo "  4. Open workspace/graphs/vla_policy_simpler.json — already configured"
echo "     for RT-1-X (model=rt1_model, policy=rt1_policy). Pick split/task/"
echo "     episode in the SIMPLER controller, set policy_setup (widowx_bridge"
echo "     vs google_robot) on the Predict node, click Play."
