"""smartway 预测器的启动垫片（放进 PYTHONPATH，解释器启动时自动执行）。

ac-wp 环境有新 torch（5090 可用）但没有 habitat_sim（编译包，装不进）。
深度编码器 VlnResnetDepthEncoder 只需要 habitat_baselines 里的 ResNet 政策
代码；把 habitat_baselines/__init__ 顺带拉进来的三个 IL trainer（其 import
链一路捅到 habitat_sim）预先用空壳顶掉——运行期根本用不到它们。
"""
import sys
import types


def _stub(name: str, **attrs) -> None:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


class _UnusedTrainer:  # 只为满足 from ... import X；永不实例化
    pass


for _name, _cls in (
    ("habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer", "EQACNNPretrainTrainer"),
    ("habitat_baselines.il.trainers.pacman_trainer", "PACMANTrainer"),
    ("habitat_baselines.il.trainers.vqa_trainer", "VQATrainer"),
):
    _stub(_name, **{_cls: _UnusedTrainer})


class _AutoModule(types.ModuleType):
    """全接受的假模块：任意属性链返回新的假模块，可调用（返回自己）。
    habitat 的 actions/registry 在 import 期只做注册，运行期没人真用
    habitat_sim —— 深度编码器是纯 torch。"""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        val = _AutoModule(self.__name__ + "." + name)
        setattr(self, name, val)
        return val

    def __call__(self, *a, **k):
        return self


# torch>=2.6 默认 weights_only=True；DDPPO 权重里存了 argparse.Namespace（本地可信文件）
try:
    import argparse
    import torch
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass

_hs = _AutoModule("habitat_sim")
# 被当作基类用的名字必须是真正的 class（模块实例没法被继承）
_hs.Simulator = type("Simulator", (), {})
_hs.Agent = type("Agent", (), {})
sys.modules.setdefault("habitat_sim", _hs)
