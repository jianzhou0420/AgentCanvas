"""
Learning rate schedulers for VLA training.

Includes OpenPI-compatible cosine decay scheduler with non-zero end learning rate.
"""

import math
from typing import Optional, Union

from diffusers.optimization import (
    TYPE_TO_SCHEDULER_FUNCTION,
    Optimizer,
    SchedulerType,
)
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    decay_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    OpenPI-compatible cosine decay schedule with warmup.

    Unlike diffusers' cosine schedule which decays to 0, this scheduler
    decays to `peak_lr * decay_lr_ratio`, matching OpenPI's behavior.

    OpenPI defaults:
        - num_warmup_steps: 1,000
        - num_decay_steps: 30,000
        - peak_lr: 2.5e-5
        - decay_lr: 2.5e-6 (i.e., decay_lr_ratio = 0.1)

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps (linear increase from 0 to peak_lr).
        num_decay_steps: Number of decay steps for cosine annealing.
        decay_lr_ratio: Ratio of end LR to peak LR. Default 0.1 matches OpenPI
                        (decay_lr=2.5e-6 / peak_lr=2.5e-5 = 0.1).
        last_epoch: The index of last epoch. Default: -1.

    Returns:
        LambdaLR scheduler that matches OpenPI's CosineDecaySchedule.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            if current_step <= 0:
                return 1.0 / (num_warmup_steps + 1)
            frac = 1 - current_step / num_warmup_steps
            return (1.0 / (num_warmup_steps + 1) - 1) * frac + 1

        # Cosine decay phase: decay from 1.0 to decay_lr_ratio
        step = min(current_step - num_warmup_steps, num_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / num_decay_steps))
        # Interpolate between decay_lr_ratio and 1.0
        return (1 - decay_lr_ratio) * cosine_decay + decay_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs,
):
    """
    Unified API to get any scheduler from its name.

    Supports OpenPI-compatible "cosine_openpi" scheduler in addition to
    all diffusers schedulers.

    Args:
        name: The name of the scheduler to use. Use "cosine_openpi" for
              OpenPI-compatible cosine decay with non-zero end LR.
        optimizer: The optimizer that will be used during training.
        num_warmup_steps: The number of warmup steps.
        num_training_steps: The number of training steps (used as decay_steps
                           for cosine_openpi).
        **kwargs: Additional arguments passed to the scheduler.
            For cosine_openpi:
                - decay_lr_ratio (float): Ratio of end LR to peak LR. Default 0.1.
    """
    # Handle OpenPI-compatible cosine scheduler
    if name == "cosine_openpi":
        if num_warmup_steps is None:
            raise ValueError("cosine_openpi requires `num_warmup_steps`")
        if num_training_steps is None:
            raise ValueError("cosine_openpi requires `num_training_steps`")

        decay_lr_ratio = kwargs.pop("decay_lr_ratio", 0.1)
        last_epoch = kwargs.pop("last_epoch", -1)

        return get_cosine_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=num_training_steps,
            decay_lr_ratio=decay_lr_ratio,
            last_epoch=last_epoch,
        )

    # Fall back to diffusers schedulers
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
