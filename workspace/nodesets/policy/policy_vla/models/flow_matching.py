"""Reusable flow matching utilities for VLAWorkspace.

Centralizes the flow matching math (time sampling, interpolation, velocity
targets, Euler ODE integration) so that policy files don't duplicate it.
"""

import torch


class FlowMatching:
    """Straight-line flow matching with Beta time sampling.

    Args:
        alpha: Beta distribution alpha parameter (default 1.5).
        beta_param: Beta distribution beta parameter (default 1.0).
        t_min: Minimum time value (default 0.001). Max is always 1.0.
    """

    def __init__(self, alpha: float = 1.5, beta_param: float = 1.0,
                 t_min: float = 0.001):
        self.alpha = alpha
        self.beta_param = beta_param
        self.t_min = t_min

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time from Beta distribution, matching Pi0/SmolVLA.

        Beta(1.5, 1.0) biases toward t~1 (noisy end) where model impact is highest.
        Maps [0,1] -> [t_min, 1.0] via: t = beta_sample * (1 - t_min) + t_min.
        """
        alpha_t = torch.as_tensor(self.alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(self.beta_param, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        return dist.sample((batch_size,)).to(device) * (1.0 - self.t_min) + self.t_min

    @staticmethod
    def interpolate(actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Straight-line interpolation: x_t = t * noise + (1 - t) * actions."""
        t = time[:, None, None]  # [B, 1, 1] for broadcasting over [B, T, D]
        return t * noise + (1 - t) * actions

    @staticmethod
    def velocity_target(actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Target velocity: u_t = noise - actions (constant along the path)."""
        return noise - actions

    @staticmethod
    def euler_sample(model_fn, noise: torch.Tensor, num_steps: int, device: torch.device) -> torch.Tensor:
        """Euler ODE integration from t=1 (noise) to t=0 (clean actions).

        Args:
            model_fn: callable(x_t, time_batch) -> v_t
            noise: [B, T, D] initial noise tensor
            num_steps: number of Euler steps
            device: torch device

        Returns:
            [B, T, D] denoised action predictions
        """
        x_t = noise
        dt = -1.0 / num_steps
        time = 1.0
        while time >= -dt / 2:
            t_batch = torch.full((noise.shape[0],), time, dtype=torch.float32, device=device)
            v_t = model_fn(x_t, t_batch)
            x_t = x_t + dt * v_t
            time += dt
        return x_t
