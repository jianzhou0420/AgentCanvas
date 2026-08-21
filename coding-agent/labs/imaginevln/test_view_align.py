"""P1 单测：视图对齐的数学性质。不碰任何服务，纯几何。

跑法：python test_view_align.py
"""
from __future__ import annotations

import math
import sys

from imagine_tools import (TURN_STEP, align_view, legacy_actions, norm_pi,
                           residual_actions)


def net_turn(actions: list[int]) -> int:
    """动作序列的净转向步数（+左 −右）。"""
    return sum(1 if a == 2 else -1 if a == 3 else 0 for a in actions)


def main() -> int:
    fails = 0

    # 1. 残差 ≤ 3 步（45°）—— 视图对齐的核心承诺
    for deg in range(-180, 181):
        acts, _dir, _view, res = residual_actions(math.radians(deg), 3.0)
        turns = sum(1 for a in acts if a in (2, 3))
        if turns > 3 or abs(res) > math.radians(45) + 1e-9:
            print(f"FAIL residual: {deg}° -> {turns} turns, res {math.degrees(res):.1f}°")
            fails += 1

    # 2. 运动学不变量：视图中心 + 残差量化 == 一期的整体量化
    #    （中心角都是 15° 的倍数，所以净转角必须逐度一致）
    for deg in range(-180, 181):
        th = math.radians(deg)
        acts_new, _d, _v, res = residual_actions(th, 3.0)
        center = align_view(th)[0]
        new_net = norm_pi(center + net_turn(acts_new) * TURN_STEP)
        old_net = norm_pi(net_turn(legacy_actions(th, 3.0)) * TURN_STEP)
        # 一期从 0 起转，净角可等价 mod 360（180° 时 +180 与 −180 同向）
        if abs(norm_pi(new_net - old_net)) > 1e-6:
            print(f"FAIL net-yaw: {deg}° -> new {math.degrees(new_net):.0f}° "
                  f"vs old {math.degrees(old_net):.0f}°")
            fails += 1

    # 3. 前进步数一致（同一个 floor）
    for dist in (0.1, 0.25, 0.9, 1.0, 2.49, 2.5, 3.01, 6.0):
        n_new = sum(1 for a in residual_actions(1.0, dist)[0] if a == 1)
        n_old = sum(1 for a in legacy_actions(1.0, dist) if a == 1)
        if n_new != n_old:
            print(f"FAIL fwd: {dist} -> {n_new} vs {n_old}")
            fails += 1

    # 4. 四个正方向落对视图、零残差
    for deg, want in ((0, "Front"), (90, "Left"), (180, "Back"), (-90, "Right"),
                      (270, "Right"), (-180, "Back")):
        _c, _d, view, res = align_view(math.radians(deg))
        if view != want or abs(res) > 1e-9:
            print(f"FAIL view: {deg}° -> {view} res {math.degrees(res):.1f}° (want {want})")
            fails += 1

    # 5. 序列长度收益：背后 3 m 的点
    old = legacy_actions(math.pi, 3.0)
    new = residual_actions(math.pi, 3.0)[0]
    print(f"Back 3.0m: legacy {len(old)} acts ({sum(1 for a in old if a in (2,3))} turns) "
          f"-> view-aligned {len(new)} acts ({sum(1 for a in new if a in (2,3))} turns)")
    assert len(new) == 12 and len(old) == 24

    # 6. 左右方向约定：+50°（左偏）在 Left 视图，残差 −40° -> 右转 3 步 (action 3)
    acts, _d, view, res = residual_actions(math.radians(50), 1.0)
    ok = view == "Left" and acts[:3] == [3, 3, 3]
    print(f"+50° -> {view} view, residual {math.degrees(res):.0f}°, acts {acts[:4]}... "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails += 1

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
