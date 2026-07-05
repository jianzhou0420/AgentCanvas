"""DROID Diffusion Policy for VLAWorkspace (Paper Table II).

Architecture aligned with DROID paper Appendix F-A:
- ResNet50 (ImageNet pretrained) + Flatten + Linear(512) per camera
- Low-dim obs: identity (pos, quat, gripper, language_distilbert)
- Concat all -> MLP(1800 -> 1024 -> 512 -> 512) per obs timestep
- global_cond = [B, 512 * n_obs_steps]
- UNet: input_dim=10, global_cond_dim=1024, dims=[256,512,1024], dsed=256
- DDIM: 100 train steps, 10 inference steps
- Language: DistilBERT -> sum pooling -> [768] injected as low_dim obs key
- noise_samples=8 for training
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import SchedulerMixin

from workspace.nodesets.policy.policy_adapter_vla.models.droid.conditional_unet1d import ConditionalUnet1D
from workspace.nodesets.policy.policy_adapter_vla.models.droid.obs_encoder import create_obs_encoder
from workspace.nodesets.policy.policy_adapter_vla.policies.base_policy import BasePolicy


class DroidDiffusionPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: SchedulerMixin,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 10,
        crop_shape=None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        obs_encoder_group_norm: bool = True,
        skip_normalization: bool = True,  # Deprecated, unused
        noise_samples: int = 8,
        use_language: bool = False,
        language_model_name: str = "distilbert-base-uncased",
        feature_dimension: int = 512,
        image_size: int = 128,
        crop_size: int = 116,
        color_jitter: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
        }
        obs_key_shapes = dict()
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

        # Store obs key shapes for time-dimension detection in predict_action
        self._obs_key_shapes = obs_key_shapes

        # Language conditioning: inject as low_dim obs key
        self.use_language = use_language
        self.language_model_name = language_model_name
        self._lang_model = None
        self._lang_tokenizer = None
        if use_language:
            obs_key_shapes["language_distilbert"] = [768]
            obs_config["low_dim"].append("language_distilbert")

        # Create observation encoder (handles resize, augmentation, MLP internally)
        obs_encoder = create_obs_encoder(
            obs_key_shapes=obs_key_shapes,
            obs_config=obs_config,
            feature_dimension=feature_dimension,
            obs_encoder_group_norm=obs_encoder_group_norm,
            image_size=image_size,
            crop_size=crop_size,
            color_jitter=color_jitter,
        )

        obs_feature_dim = obs_encoder.output_shape()[0]  # 512 (after MLP)

        # Global conditioning dim = obs_features * n_obs_steps
        global_cond_dim = obs_feature_dim * n_obs_steps

        # Create DROID's UNet
        model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.noise_samples = noise_samples

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def _get_lang_model(self, device):
        """Lazy-load DistilBERT tokenizer + model (frozen, no grad)."""
        if self._lang_model is None:
            from transformers import AutoModel, AutoTokenizer

            self._lang_tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
            self._lang_model = AutoModel.from_pretrained(self.language_model_name)
            self._lang_model = self._lang_model.to(device)
            self._lang_model.eval()
            for p in self._lang_model.parameters():
                p.requires_grad = False
        return self._lang_tokenizer, self._lang_model

    def _encode_language(self, prompts, device):
        """Encode prompt strings to language embeddings.

        Uses sum pooling over last_hidden_state (matching paper).

        Args:
            prompts: List of strings
            device: Target device

        Returns:
            [B, 768] summed token embeddings
        """
        tokenizer, lang_model = self._get_lang_model(device)
        with torch.no_grad():
            tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
                device
            )
            outputs = lang_model(**tokens)
            # Sum pooling over sequence dimension (paper: .last_hidden_state.sum(dim=1))
            lang_emb = outputs.last_hidden_state.sum(dim=1)  # [B, 768]
        return lang_emb.float()

    def _inject_language(self, nobs, prompts, device):
        """Encode prompts and inject as obs key 'language_distilbert'.

        Expands [B, 768] -> [B, T, 768] to match obs time dimension.
        """
        if not self.use_language or prompts is None:
            return
        if not (isinstance(prompts, (list, tuple)) and len(prompts) > 0):
            return

        lang_emb = self._encode_language(prompts, device)  # [B, 768]
        # Determine time dimension from existing obs
        T = next(iter(nobs.values())).shape[1]
        lang_emb = lang_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, 768]
        nobs["language_distilbert"] = lang_emb

    def _encode_obs(self, nobs, batch_size, device):
        """Encode observations to global conditioning vector.

        Args:
            nobs: Dict of observation tensors with shape [B, T, ...]
            batch_size: B
            device: Target device

        Returns:
            [B, obs_feature_dim * n_obs_steps] obs features
        """
        # Slice to n_obs_steps and reshape to [B*To, ...]
        this_nobs = {}
        for k, v in nobs.items():
            v = v[:, : self.n_obs_steps]
            this_nobs[k] = v.reshape(-1, *v.shape[2:])

        obs_features = self.obs_encoder(this_nobs)  # [B*To, 512]
        obs_cond = obs_features.reshape(batch_size, -1)  # [B, 512 * To]
        return obs_cond

    # ========= training ============
    def compute_loss(self, batch):
        """Compute diffusion training loss with noise_samples support."""
        nobs = batch["obs"]
        nactions = batch["action"]

        B = nactions.shape[0]
        device = nactions.device

        # Inject language as obs key
        prompts = batch.get("prompt")
        self._inject_language(nobs, prompts, device)

        # Encode observations (language included via obs encoder MLP)
        obs_cond = self._encode_obs(nobs, B, device)  # [B, 512 * To]

        # Sample noise(s) to add to actions
        noise = torch.randn([self.noise_samples] + list(nactions.shape), device=device)

        # Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # Add noise (forward diffusion) - support multiple noise samples
        noisy_actions = torch.cat(
            [
                self.noise_scheduler.add_noise(nactions, noise[i], timesteps)
                for i in range(self.noise_samples)
            ],
            dim=0,
        )

        # Repeat conditioning for noise_samples
        obs_cond_rep = obs_cond.repeat(self.noise_samples, 1)
        timesteps_rep = timesteps.repeat(self.noise_samples)

        # Predict noise
        noise_pred = self.model(noisy_actions, timesteps_rep, global_cond=obs_cond_rep)

        # L2 loss
        noise_flat = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
        loss = F.mse_loss(noise_pred, noise_flat)

        return loss

    # ========= inference ============
    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict actions using DDIM/DDPM denoising."""
        # Extract obs sub-dict if wrapped
        if "obs" in obs_dict and isinstance(obs_dict["obs"], dict):
            raw_obs = obs_dict["obs"]
        else:
            raw_obs = obs_dict

        nobs = raw_obs

        value = next(iter(nobs.values()))
        B = value.shape[0]
        device = value.device

        # Add time dimension if missing (rollout envs provide single frames
        # [B, ...] but _encode_obs expects [B, T, ...]).
        # Repeat the single frame to fill n_obs_steps.
        for k, v in nobs.items():
            if isinstance(v, torch.Tensor) and k in self._obs_key_shapes:
                expected_ndim = len(self._obs_key_shapes[k]) + 2  # batch + time + shape
                if v.ndim == expected_ndim - 1:
                    nobs = {
                        kk: vv.unsqueeze(1).expand(-1, self.n_obs_steps, *vv.shape[1:]).contiguous()
                        if isinstance(vv, torch.Tensor)
                        else vv
                        for kk, vv in nobs.items()
                    }
                    break

        # Inject language as obs key
        prompts = obs_dict.get("prompt")
        self._inject_language(nobs, prompts, device)

        # Encode observations
        obs_cond = self._encode_obs(nobs, B, device)

        # Initialize from Gaussian noise
        noisy_action = torch.randn((B, self.horizon, self.action_dim), device=device)

        # Denoising loop (works with both DDPM and DDIM schedulers)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.model(
                sample=noisy_action,
                timestep=t,
                global_cond=obs_cond,
            )
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample

        # Extract action window (matching DROID: start=To-1, end=start+Ta)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = noisy_action[:, start:end]

        return {
            "action": action,
            "action_pred": noisy_action,
        }

    # ========= optimizer & scheduler ============
    def get_optimizer(self, lr=1e-4, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def get_scheduler(self, optimizer, **kwargs):
        """Constant LR (no decay) — matching paper."""
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


# ───── DEFAULTS (transcribed from droid_dp_libero.yaml) ─────
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
        "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "prediction_type": "epsilon",
    },
    "horizon": 16,
    "n_action_steps": 8,
    "n_obs_steps": 2,
    "num_inference_steps": 10,
    "diffusion_step_embed_dim": 256,
    "down_dims": [256, 512, 1024],
    "kernel_size": 5,
    "n_groups": 8,
    "obs_encoder_group_norm": True,
    "skip_normalization": True,
    "noise_samples": 8,
    "use_language": True,
    "language_model_name": "distilbert-base-uncased",
    "feature_dimension": 512,
    "image_size": 128,
    "crop_size": 116,
    "color_jitter": True,
}
