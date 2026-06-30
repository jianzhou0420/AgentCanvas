from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from workspace.nodesets.policy.policy_vla.models.dp.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)
from workspace.nodesets.policy.policy_vla.models.dp.diffusion.mask_generator import (
    LowdimMaskGenerator,
)
from workspace.nodesets.policy.policy_vla.models.dp.vision import (
    create_obs_encoder,
)
from workspace.nodesets.policy.policy_vla.models.z_utils.pytorch_util import dict_apply
from workspace.nodesets.policy.policy_vla.policies.base_policy import BasePolicy


class DiffusionUnetHybridImagePolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        rot_aug=False,
        skip_normalization=False,  # Deprecated, unused
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        # modified to support no obs keys
        if obs_shape_meta is not None:
            for key, attr in obs_shape_meta.items():
                shape = attr["shape"]
                obs_key_shapes[key] = list(shape)

                type = attr.get("type", "low_dim")
                if type == "rgb":
                    obs_config["rgb"].append(key)
                elif type == "low_dim":
                    obs_config["low_dim"].append(key)
                else:
                    raise RuntimeError(f"Unsupported obs type: {type}")
        # end of parsing modification

        # Apply crop_shape to RGB observations
        # Cropping is handled by the adaptor transform pipeline, but the encoder
        # needs to know the cropped input size
        if crop_shape is not None:
            crop_h, crop_w = crop_shape
            for key in obs_config["rgb"]:
                if key in obs_key_shapes:
                    c = obs_key_shapes[key][0]  # channels
                    obs_key_shapes[key] = [c, crop_h, crop_w]

        # Create observation encoder using local implementation
        obs_encoder = create_obs_encoder(
            obs_key_shapes=obs_key_shapes,
            obs_config=obs_config,
            feature_dimension=64,
            obs_encoder_group_norm=obs_encoder_group_norm,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from .ckpt file.

        DP checkpoint keys (model.*, obs_encoder.*) match policy attributes directly
        — no prefix stripping needed.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet

        # Extract obs sub-dict if wrapped in {'obs': {...}, ...} format
        # (PolicyServer passes wrapped format; runner passes flat obs dict)
        if "obs" in obs_dict and isinstance(obs_dict["obs"], dict):
            raw_obs = obs_dict["obs"]
        else:
            raw_obs = obs_dict

        nobs = raw_obs
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

            # DEBUG: Log observation encoder input on first call
            if not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                print("=== DEBUG: Observation Encoder Input ===")
                for k, v in this_nobs.items():
                    print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

            nobs_features = self.obs_encoder(this_nobs)

            # DEBUG: Log encoder output
            if hasattr(self, "_debug_logged") and self._debug_logged:
                print("=== DEBUG: Encoder Output ===")
                print(
                    f"  nobs_features: shape={nobs_features.shape}, range=[{nobs_features.min():.4f}, {nobs_features.max():.4f}]"
                )
                print(
                    f"  First sample features[:10]: {nobs_features[0, :10].detach().cpu().numpy()}"
                )

            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)

            # DEBUG: Log global conditioning
            if hasattr(self, "_debug_logged") and self._debug_logged:
                print("=== DEBUG: Global Conditioning ===")
                print(
                    f"  global_cond: shape={global_cond.shape}, range=[{global_cond.min():.4f}, {global_cond.max():.4f}]"
                )
                self._debug_logged = False  # Only log once
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask, global_cond=global_cond, **self.kwargs
        )

        action_pred = nsample[..., :Da]

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def get_optimizer(self, lr=1e-4, weight_decay=1e-6, betas=(0.95, 0.999), **kwargs):
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
        )

    def get_scheduler(self, optimizer, num_training_steps, last_epoch=-1, **kwargs):
        from workspace.nodesets.policy.policy_vla.models.dp.common.lr_scheduler import get_scheduler

        return get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = batch["obs"]
        nactions = batch["action"]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            # modified to support n_obs_steps
            if len(nobs) == 0:
                global_cond = torch.empty(
                    (batch_size, 0), device=nactions.device, dtype=nactions.dtype
                )
            else:
                this_nobs = dict_apply(
                    nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
                )
                nobs_features = self.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
                if not hasattr(self, "_logged_global_cond_dim"):
                    self._logged_global_cond_dim = True
                    print(f"[DEBUG dp_c] global_cond shape: {global_cond.shape}")
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape, device=trajectory.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss


# ───── DEFAULTS (transcribed from diffusion_policy_libero.yaml) ─────
# `shape_meta` is the resolved upstream `${IO_meta.shape_meta}` interpolation.
# `noise_scheduler` is a Hydra `_target_:` spec instantiated by the canvas
# loader (see policy_vla.__init__._instantiate_specs) before constructing the policy.
DEFAULT_KWARGS: dict = {
    "shape_meta": {
        "obs": {
            "agentview_image": {"shape": [3, 256, 256], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 256, 256], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [1], "type": "low_dim"},
        },
        "action": {"shape": [10]},
    },
    "noise_scheduler": {
        "_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "variance_type": "fixed_small",
        "clip_sample": True,
        "prediction_type": "epsilon",
    },
    "horizon": 16,
    "n_action_steps": 8,
    "n_obs_steps": 2,
    "num_inference_steps": 100,
    "obs_as_global_cond": True,
    "crop_shape": [76, 76],
    "diffusion_step_embed_dim": 128,
    "down_dims": [512, 1024, 2048],
    "kernel_size": 5,
    "n_groups": 8,
    "cond_predict_scale": True,
    "obs_encoder_group_norm": True,
    "skip_normalization": True,
}
