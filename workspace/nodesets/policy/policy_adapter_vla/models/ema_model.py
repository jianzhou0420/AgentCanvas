import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    Exponential Moving Average of models weights.

    Supports two modes:
    1. Fixed decay (OpenPI-compatible): Set `fixed_decay` to a value like 0.99
    2. Scheduled decay (diffusion-style): Uses power-law warmup schedule
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        fixed_decay=None,
    ):
        """
        Args:
            model: The model to track with EMA.
            update_after_step (int): Start EMA updates after this step. Default: 0.
            fixed_decay (float, optional): If set, uses a constant decay factor (OpenPI-style).
                OpenPI uses 0.99 for full fine-tuning. When set, ignores inv_gamma/power/min/max.
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
                Only used when fixed_decay is None.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
                Only used when fixed_decay is None.
                @crowsonkb's notes on EMA Warmup:
                - gamma=1, power=2/3: good for 1M+ steps (decay=0.999 at 31.6K, 0.9999 at 1M)
                - gamma=1, power=3/4: good for shorter training (decay=0.999 at 10K, 0.9999 at 215K)
            min_value (float): The minimum EMA decay rate. Default: 0.
            max_value (float): The maximum EMA decay rate. Default: 0.9999.
        """
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.fixed_decay = fixed_decay  # OpenPI-compatible fixed decay

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.

        If fixed_decay is set (OpenPI-style), returns that constant value.
        Otherwise, uses power-law warmup schedule.
        """
        # OpenPI-compatible fixed decay
        if self.fixed_decay is not None:
            step = max(0, optimization_step - self.update_after_step - 1)
            if step <= 0:
                return 0.0
            return self.fixed_decay

        # Scheduled decay (diffusion-style warmup)
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules(), strict=False
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=False
            ):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1
