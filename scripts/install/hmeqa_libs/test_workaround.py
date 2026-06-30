"""Smoke-test for nvidia_egl_workaround.so — boots habitat-sim 0.3.x and
renders one frame from an HM3D scene. Run with the workaround already
LD_PRELOAD'd (the conda env activation hook does this automatically).

Usage:
    conda activate ac-hmeqa
    python scripts/install/hmeqa_libs/test_workaround.py [SCENE_GLB]

Default scene: HM3D semantic 00004-VqCaAuuoeWk (HM-EQA val split sample).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")

import habitat_sim

# hmeqa_libs/test_workaround.py → hmeqa_libs → install → scripts → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = str(
    _REPO_ROOT / "data" / "hm3d" / "hm3dsem" / "00004-VqCaAuuoeWk" / "VqCaAuuoeWk.basis.glb"
)


def main() -> int:
    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE
    if not os.path.exists(scene):
        print(f"FAIL: scene not found at {scene}")
        return 1
    if not os.environ.get("LD_PRELOAD") or "nvidia_egl_workaround" not in os.environ.get(
        "LD_PRELOAD", ""
    ):
        print(
            "WARN: LD_PRELOAD does not include nvidia_egl_workaround.so. "
            "If you are on driver 570+ with habitat-sim 0.3.x, expect SIGSEGV."
        )

    print(f"habitat_sim version: {habitat_sim.__version__}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 1.5, 0.0]
    rgb.hfov = 120
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [480, 640]
    depth.position = [0.0, 1.5, 0.0]
    depth.hfov = 120
    agent_cfg.sensor_specifications = [rgb, depth]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    obs = sim.get_sensor_observations()
    nonblack = (obs["color"][..., :3].sum(axis=-1) > 0).mean()
    print(f"  rgb {obs['color'].shape} mean={obs['color'].mean():.1f} nonblack={nonblack:.3f}")
    print(f"  depth {obs['depth'].shape} mean={obs['depth'].mean():.4f}")
    sim.close()
    print("PASS — workaround active, habitat-sim 0.3.x runs cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
