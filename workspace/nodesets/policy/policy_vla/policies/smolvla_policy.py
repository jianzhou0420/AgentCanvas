"""SmolVLA Policy wrapper for VLAWorkspace (original cross-attention mode).

Wraps LeRobot's SmolVLAPolicy to implement VLAWorkspace's BasePolicy interface.
This enables training and inference of SmolVLA within VLAWorkspace's trainer.py
pipeline without modifying any SmolVLA source code.

For the FiLM-conditioning variant, see smolvla_policy_film.py.
"""

from __future__ import annotations

import logging

import torch

from workspace.nodesets.policy.policy_vla.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class SmolVLAPolicy(BasePolicy):
    """VLAWorkspace wrapper around LeRobot's SmolVLAPolicy (cross-attention mode).

    Args:
        image_features: Dict mapping image key names to shape lists,
            e.g. {"observation.images.front": [3, 256, 256]}.
        state_dim: Actual state dimension (before padding).
        action_dim: Actual action dimension (before padding).
        chunk_size: Number of action steps per prediction.
        n_action_steps: Number of action steps to execute.
        num_steps: Number of denoising steps for inference.
        vlm_model_name: HuggingFace model name for the VLM backbone.
        pretrained_path: Path or HF repo for pretrained weights.
        freeze_vision_encoder: Whether to freeze the vision encoder.
        train_expert_only: Whether to only train the expert + projections.
        load_vlm_weights: Whether to load pretrained VLM weights.
    """

    def __init__(
        self,
        image_features: dict,
        state_dim: int = 8,
        action_dim: int = 7,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        num_steps: int = 10,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        pretrained_path: str | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        load_vlm_weights: bool = True,
        **kwargs,
    ):
        super().__init__()

        from lerobot.configs.types import FeatureType, PolicyFeature

        from workspace.nodesets.policy.policy_vla.models.smolvla.configuration_smolvla import (
            SmolVLAConfig,
        )
        from workspace.nodesets.policy.policy_vla.models.smolvla.modeling_smolvla import (
            SmolVLAPolicy as LeRobotSmolVLAPolicy,
        )

        # Build input/output feature dicts for SmolVLAConfig
        input_features = {}
        for key, shape in image_features.items():
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=tuple(shape))
        input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        )

        output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

        config = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            num_steps=num_steps,
            vlm_model_name=vlm_model_name,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            load_vlm_weights=load_vlm_weights,
        )

        if pretrained_path:
            logger.info(f"Loading SmolVLA from pretrained: {pretrained_path}")
            self._smolvla = LeRobotSmolVLAPolicy.from_pretrained(pretrained_path, config=config)
        else:
            logger.info("Creating SmolVLA from scratch")
            self._smolvla = LeRobotSmolVLAPolicy(config)

        # Note: SmolVLA handles its own parameter freezing internally via
        # SmolVLMWithExpertModel.set_requires_grad() and VLAFlowMatching.set_requires_grad()
        # based on freeze_vision_encoder, train_expert_only, and train_state_proj config flags.
        # We do NOT apply additional freezing here to avoid overriding SmolVLA's logic
        # (e.g., it explicitly freezes lm_expert.lm_head to avoid unused param issues).

        # Log parameter counts
        total_params = sum(p.numel() for p in self._smolvla.parameters())
        trainable_params = sum(p.numel() for p in self._smolvla.parameters() if p.requires_grad)
        logger.info(
            f"SmolVLA parameters: {trainable_params:,} trainable / {total_params:,} total "
            f"({trainable_params / total_params * 100:.1f}%)"
        )

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint from .ckpt file.

        Checkpoint keys (_smolvla.*) match this wrapper's state_dict directly.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.info(f"Loaded checkpoint (step={ckpt.get('global_step', 'unknown')})")

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute training loss.

        SmolVLA's forward() returns (loss, loss_dict). We extract the scalar loss.
        """
        loss, _loss_dict = self._smolvla.forward(batch)
        return loss

    def predict_action(self, batch: dict) -> torch.Tensor:
        """Predict action chunk from observations.

        Returns:
            Action tensor of shape (B, chunk_size, action_dim).
        """
        with torch.no_grad():
            actions = self._smolvla.predict_action_chunk(batch)
        return actions

    def get_scheduler(self, optimizer, num_training_steps, last_epoch=-1, **kwargs):
        scheduler_config = self._smolvla.config.get_scheduler_preset()
        return scheduler_config.build(optimizer, num_training_steps)

    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        """Return AdamW optimizer using SmolVLA's config presets.

        kwargs from YAML optimizer config can override any preset value.
        """
        cfg = self._smolvla.config
        trainable_params = [p for p in self._smolvla.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=kwargs.get("lr", cfg.optimizer_lr),
            betas=kwargs.get("betas", cfg.optimizer_betas),
            eps=kwargs.get("eps", cfg.optimizer_eps),
            weight_decay=kwargs.get("weight_decay", cfg.optimizer_weight_decay),
        )

    def reset(self) -> None:
        """Reset action queue for new episode."""
        self._smolvla.reset()


# ───── DEFAULTS (transcribed from smolvla_libero.yaml) ─────
DEFAULT_KWARGS: dict = {
    "image_features": {
        "observation.images.front": [3, 256, 256],
        "observation.images.wrist": [3, 256, 256],
    },
    "state_dim": 8,
    "action_dim": 7,
    "chunk_size": 50,
    "n_action_steps": 50,
    "num_steps": 10,
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "pretrained_path": "lerobot/smolvla_base",
    "freeze_vision_encoder": True,
    "train_expert_only": True,
    "load_vlm_weights": True,
}
