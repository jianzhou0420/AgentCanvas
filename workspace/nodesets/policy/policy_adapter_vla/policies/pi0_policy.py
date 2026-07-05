"""
Pi0 Policy for TaskFusion.

A clean policy wrapper for the Pi0 Vision-Language-Action model.
This policy expects pre-processed data from an Adaptor (e.g., Pi0Adaptor).

Data Flow:
    LeRobot Dataset Output
        ↓
    Adaptor (Pi0Adaptor: normalize, tokenize, pad, remap)
        ↓
    Pi0Observation (ready for model)
        ↓
    Pi0Policy.compute_loss() or Pi0Policy.predict_action()

Reference: OpenPi's Pi0 implementation
    - models/pi0.py: Pi0 model architecture
    - models/model.py: Observation dataclass
    - models/pi0_config.py: Configuration
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import torch

from workspace.nodesets.policy.policy_adapter_vla.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


# =============================================================================
# Pi0 Observation (Model Input Format)
# =============================================================================


@dataclass
class Pi0Observation:
    """
    Observation format expected by Pi0 model.

    This should be produced by a DataPreprocess class before passing to the policy.

    Attributes:
        images: Dict of images, each float32 [B, H, W, 3] in range [-1, 1]
            Required keys: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        image_masks: Dict of bool [B] indicating valid images
        state: float32 [B, 32] - padded and normalized state
        tokenized_prompt: int64 [B, max_token_len] - tokenized language instruction
        tokenized_prompt_mask: bool [B, max_token_len] - attention mask for tokens
    """

    images: dict[str, torch.Tensor]  # {camera_name: [B, H, W, 3]}
    image_masks: dict[str, torch.Tensor]  # {camera_name: [B]}
    state: torch.Tensor  # [B, 32]
    tokenized_prompt: torch.Tensor  # [B, max_token_len]
    tokenized_prompt_mask: torch.Tensor  # [B, max_token_len]

    # For Pi0-FAST model (optional)
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Pi0Observation:
        """Create Pi0Observation from a dictionary (output of DataPreprocess)."""
        # Convert uint8 images to float32 [-1, 1] and HWC to CHW for PyTorch
        # NOTE: Original OpenPi has different paths for JAX (HWC) vs PyTorch (CHW).
        # Since we use PyTorch only, we always convert to CHW format.
        images = {}
        for key, img in data["images"].items():
            if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                # PyTorch uint8: convert to float32 CHW (matches original OpenPi)
                images[key] = img.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            elif isinstance(img, np.ndarray) and img.dtype == np.uint8:
                # NumPy uint8: convert to torch float32 CHW
                img_float = torch.from_numpy(img).to(torch.float32)
                images[key] = img_float.permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            else:
                # Already float, assume already in correct format
                images[key] = img if isinstance(img, torch.Tensor) else torch.from_numpy(img)

        # Convert image masks
        image_masks = {}
        for key, mask in data["image_masks"].items():  # changed from data["image_mask"]
            if isinstance(mask, torch.Tensor):
                image_masks[key] = mask
            else:
                image_masks[key] = torch.from_numpy(np.asarray(mask))

        return cls(
            images=images,
            image_masks=image_masks,
            state=data["state"]
            if isinstance(data["state"], torch.Tensor)
            else torch.from_numpy(data["state"]),
            tokenized_prompt=data["tokenized_prompt"]
            if isinstance(data["tokenized_prompt"], torch.Tensor)
            else torch.from_numpy(data["tokenized_prompt"]),
            tokenized_prompt_mask=data["tokenized_prompt_mask"]
            if isinstance(data["tokenized_prompt_mask"], torch.Tensor)
            else torch.from_numpy(data["tokenized_prompt_mask"]),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> Pi0Observation:
        """Move all tensors to device and optionally cast floating-point tensors to dtype.

        Args:
            device: Target device for all tensors.
            dtype: Optional dtype for floating-point tensors (images, state).
                   Non-floating-point tensors (masks, tokens) are not cast.
        """

        def convert(t: torch.Tensor) -> torch.Tensor:
            """Move to device and cast to dtype if floating-point."""
            t = t.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            return t

        return Pi0Observation(
            images={k: convert(v) for k, v in self.images.items()},
            image_masks={k: v.to(device) for k, v in self.image_masks.items()},
            state=convert(self.state),
            tokenized_prompt=self.tokenized_prompt.to(device),
            tokenized_prompt_mask=self.tokenized_prompt_mask.to(device),
            token_ar_mask=self.token_ar_mask.to(device) if self.token_ar_mask is not None else None,
            token_loss_mask=self.token_loss_mask.to(device)
            if self.token_loss_mask is not None
            else None,
        )


# =============================================================================
# Pi0 Configuration
# =============================================================================


@dataclass
class Pi0Config:
    """
    Configuration for Pi0 model.

    Reference: openpi/models/pi0_config.py
    """

    # Model architecture
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # Dimensions (Pi0 uses 32 internally, padded from actual action dim)
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # Pi0.5 specific
    pi05: bool = False
    discrete_state_input: bool = False

    # Image settings
    image_resolution: tuple = (224, 224)
    image_keys: tuple = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __post_init__(self):
        if self.pi05 and self.max_token_len == 48:
            self.max_token_len = 200
        if self.discrete_state_input is False and self.pi05:
            self.discrete_state_input = True


# =============================================================================
# Pi0 Policy
# =============================================================================


class Pi0Policy(BasePolicy):
    """
    Pi0 Vision-Language-Action Policy.

    This policy expects pre-processed data from an Adaptor (e.g., Pi0Adaptor).
    All data transformations (normalize, tokenize, pad) should be done by
    the adaptor before passing data to this policy.

    Expected input format (from Pi0Adaptor):
        {
            "image": {
                "base_0_rgb": float32 [B, C, H, W] in [-1, 1],
                "left_wrist_0_rgb": float32 [B, C, H, W] in [-1, 1],
                "right_wrist_0_rgb": float32 [B, C, H, W] in [-1, 1],
            },
            "image_mask": {
                "base_0_rgb": bool [B],
                "left_wrist_0_rgb": bool [B],
                "right_wrist_0_rgb": bool [B],
            },
            "state": float32 [B, 32],  # Padded, normalized
            "actions": float32 [B, action_horizon, 32],  # Padded, normalized (training only)
            "tokenized_prompt": int64 [B, max_token_len],
            "tokenized_prompt_mask": bool [B, max_token_len],
        }

    Args:
        config: Pi0Config with model settings
        num_inference_steps: Number of denoising steps during inference
        use_pretrained_weight: Whether to load pretrained weights
        pretrained_weight_name: Name of pretrained checkpoint
        cache_dir: Directory to cache downloaded checkpoints
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """

    GCS_BASE = "gs://openpi-assets/checkpoints"
    CHECKPOINT_CONFIGS = {
        "pi0_base": "pi0_libero",
        "pi0_fast_base": "pi0_fast_libero",
        "pi05_base": "pi05_libero",
    }

    def __init__(
        self,
        config: Pi0Config | None = None,
        num_inference_steps: int = 10,
        use_pretrained_weight: bool = True,
        pretrained_weight_name: str | None = None,
        cache_dir: str = "data/models",
        gradient_checkpointing: bool = True,
        _enable_compile: bool = True,  # Internal: set False in subclasses that modify model after init
    ):
        super().__init__()  # Initialize BasePolicy
        self.config = config or Pi0Config()
        self.num_inference_steps = num_inference_steps
        self.cache_dir = os.path.expanduser(cache_dir)

        # Import model
        from workspace.nodesets.policy.policy_adapter_vla.models.openpi.models_pytorch.pi0_pytorch import (
            PI0Pytorch,
        )

        # Create Pi0 model
        self.model = PI0Pytorch(self.config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(f"Pi0 model created (pi05={self.config.pi05})")

        # Load pretrained weights if enabled
        if use_pretrained_weight:
            checkpoint_name = pretrained_weight_name or (
                "pi05_base" if self.config.pi05 else "pi0_base"
            )
            local_checkpoint_path = self._prepare_pretrained_checkpoint(checkpoint_name)
            self._load_pretrained_weights(local_checkpoint_path)
            logger.info(f"Loaded pretrained weights from {local_checkpoint_path}")

        # Enable torch.compile (subclasses like LoRA disable this and call later)
        if _enable_compile:
            self.model.enable_compile()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Pi0 params: total={total_params:,}, trainable={trainable_params:,}")

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def _prepare_pretrained_checkpoint(self, checkpoint_name: str) -> str:
        """
        Prepare pretrained checkpoint, downloading from GCS if needed.

        Args:
            checkpoint_name: Name of checkpoint (e.g., "pi0_base", "pi05_base")

        Returns:
            Path to local PyTorch checkpoint directory
        """
        config_name = self.CHECKPOINT_CONFIGS.get(checkpoint_name, checkpoint_name)

        # Check for existing PyTorch checkpoint
        local_pytorch_path = os.path.join(self.cache_dir, f"{checkpoint_name}_pytorch")
        model_safetensors_path = os.path.join(local_pytorch_path, "model.safetensors")

        if os.path.exists(model_safetensors_path):
            logger.info(f"Found cached PyTorch checkpoint at {local_pytorch_path}")
            return local_pytorch_path

        # Check for JAX checkpoint
        local_jax_path = os.path.join(self.cache_dir, checkpoint_name)
        params_dir = os.path.join(local_jax_path, "params")

        if not os.path.exists(params_dir):
            gcs_path = f"{self.GCS_BASE}/{checkpoint_name}/params"
            logger.info(f"Downloading checkpoint from {gcs_path}...")
            from workspace.nodesets.policy.policy_adapter_vla.models.openpi.shared.download import (
                maybe_download,
            )

            local_path = maybe_download(gcs_path)
            local_jax_path = str(local_path.parent)
            logger.info(f"Downloaded JAX checkpoint to {local_jax_path}")

        # Convert JAX to PyTorch
        logger.info("Converting JAX checkpoint to PyTorch...")
        self._convert_jax_to_pytorch(local_jax_path, local_pytorch_path, config_name)
        logger.info(f"Conversion complete: {local_pytorch_path}")

        return local_pytorch_path

    def _convert_jax_to_pytorch(self, jax_checkpoint_path: str, output_path: str, config_name: str):
        """Convert JAX checkpoint to PyTorch format."""
        if os.path.exists(os.path.join(output_path, "model.safetensors")):
            logger.info(f"Converted checkpoint already exists at {output_path}")
            return

        cmd = [
            sys.executable,
            "-m",
            "workspace.nodesets.policy.policy_adapter_vla.models.convert_jax_model_to_pytorch",
            "--checkpoint_dir",
            jax_checkpoint_path,
            "--config_name",
            config_name,
            "--output_path",
            output_path,
            "--precision",
            "bfloat16",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"Conversion failed: {error_msg}")
            raise RuntimeError(f"JAX to PyTorch conversion failed: {error_msg}")

    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights from safetensors checkpoint."""
        import safetensors.torch

        model_path = os.path.join(checkpoint_path, "model.safetensors")
        # load_model with strict=True (default) ensures:
        # - All model keys exist in checkpoint
        # - No unexpected keys in checkpoint
        # - Tied weights handled via metadata
        safetensors.torch.load_model(self.model, model_path)
        logger.info(f"Loaded pretrained weights from {model_path}")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import safetensors.torch

        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.safetensors")
        safetensors.torch.save_model(self.model, model_path)
        logger.info(f"Saved checkpoint to {model_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint from various formats.

        Supported formats:
            - Directory containing model.safetensors (OpenPi pretrained format)
            - Single .safetensors file (OpenPi finetuned format)
            - Lightning .ckpt file (VLAWorkspace training format)
        """
        if os.path.isdir(path):
            import safetensors.torch

            model_path = os.path.join(path, "model.safetensors")
            safetensors.torch.load_model(self.model, model_path)
            logger.info(f"Loaded checkpoint from {model_path}")
        elif path.endswith(".safetensors"):
            import safetensors.torch

            safetensors.torch.load_model(self.model, path)
            logger.info(f"Loaded safetensors checkpoint from {path}")
        elif path.endswith(".ckpt"):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" not in ckpt:
                raise ValueError(
                    f"Invalid Lightning checkpoint: missing 'state_dict' key in {path}"
                )
            state_dict = ckpt["state_dict"]
            # Strip 'model.' prefix — trainer saves policy.state_dict() which has model.* keys
            new_state_dict = {
                (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()
            }
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            is_lora = ckpt.get("is_lora_checkpoint", False)
            if missing and not is_lora:
                logger.warning(f"Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")
            logger.info(f"Loaded .ckpt checkpoint (step={ckpt.get('global_step', 'unknown')})")
        else:
            raise ValueError(
                f"Unknown checkpoint format: {path}. Expected directory, .safetensors, or .ckpt"
            )

    # =========================================================================
    # Training
    # =========================================================================

    def get_optimizer(self, lr=2.5e-5, weight_decay=1e-10, betas=(0.9, 0.95), **kwargs):
        return torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
        )

    def get_scheduler(self, optimizer, num_training_steps, last_epoch=-1, **kwargs):
        """OpenPi-compatible cosine decay LR schedule.

        Matches openpi/scripts/train_pytorch.py lr_schedule() exactly:
        - Warmup: linear from peak_lr/(warmup_steps+1) to peak_lr
        - Cosine decay: from peak_lr to decay_lr over (decay_steps - warmup_steps) steps

        kwargs (passed from training config):
            warmup_steps: Number of warmup steps (default: 1000)
            decay_steps: Total decay steps including warmup (default: num_training_steps)
            decay_lr_ratio: End LR as fraction of peak LR (default: 0.1)
        """
        import math

        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = kwargs.get("warmup_steps", 1000)
        decay_steps = kwargs.get("decay_steps") or num_training_steps
        decay_lr_ratio = kwargs.get("decay_lr_ratio", 0.1)

        def lr_lambda(step):
            if step < warmup_steps:
                # Match OpenPi: start from peak_lr / (warmup_steps + 1)
                init_value = 1.0 / (warmup_steps + 1)
                return init_value + (1.0 - init_value) * step / warmup_steps
            # Cosine decay — denominator is (decay_steps - warmup_steps), matching OpenPi
            progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            return decay_lr_ratio + (1.0 - decay_lr_ratio) * cos

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute loss from a pre-processed batch dictionary.

        Args:
            batch: Dict with pre-processed data from DataPreprocess
                Required keys: image, image_mask, state, actions,
                               tokenized_prompt, tokenized_prompt_mask

        Returns:
            Scalar loss tensor
        """
        # Get device and dtype from model parameters
        # The model runs in bfloat16 (AMP), so we need to cast inputs to match
        param = next(self.model.parameters())
        device = param.device
        dtype = param.dtype

        observation = Pi0Observation.from_dict(batch)
        observation = observation.to(device, dtype=dtype)

        actions = batch["actions"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions)
        actions = actions.to(device=device, dtype=dtype)

        loss = self.model(observation, actions)
        return loss.mean()

    # =========================================================================
    # Inference
    # =========================================================================

    def predict_action(self, batch: dict) -> torch.Tensor:
        """
        Predict actions from a pre-processed batch dictionary.

        Args:
            batch: Dict with pre-processed data from DataPreprocess

        Returns:
            actions: float32 [B, action_horizon, 32] - normalized actions
        """
        # Get device and dtype from model parameters
        param = next(self.model.parameters())
        device = param.device
        dtype = param.dtype

        observation = Pi0Observation.from_dict(batch)
        observation = observation.to(device, dtype=dtype)

        with torch.no_grad():
            actions = self.model.sample_actions(
                device=device,
                observation=observation,
                num_steps=self.num_inference_steps,
            )

        return actions

    # =========================================================================
    # Utilities (inherited from nn.Module: to, train, eval, parameters, state_dict, load_state_dict)
    # =========================================================================


# ───── DEFAULTS (transcribed from CORL_pi0_libero_lerobot.yaml) ─────
# Pi0Config fields (dtype=bfloat16, paligemma_variant=gemma_2b, action_expert_variant=gemma_300m,
# action_dim=32, action_horizon=50, max_token_len=48, pi05=False) all match Pi0Config() defaults,
# so the `config` key is omitted — Pi0Policy() constructs a default Pi0Config when None.
DEFAULT_KWARGS: dict = {
    "num_inference_steps": 10,
    "use_pretrained_weight": True,
    "pretrained_weight_name": None,
    "cache_dir": "data/models",
    "gradient_checkpointing": True,
}
