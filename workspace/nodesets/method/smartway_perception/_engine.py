"""PerceptionEngine — RAM+ singleton model loader.

Mirrors ``opennav_perception.PerceptionEngine._ensure_ram`` but uses the
RAM+ (Plus) variant per upstream SmartWay base_il_trainer.py:368-371.
All heavy imports deferred to first ``tag`` call so the nodeset can
register cheaply.
"""

from __future__ import annotations

import base64
import io
import logging
import threading

import numpy as np

from . import SMARTWAY_RAM_PLUS_CKPT_DEFAULT

log = logging.getLogger("agentcanvas.smartway_perception.engine")


def decode_rgb_b64(b64: str) -> np.ndarray:
    """Decode a base64 RGB image string → (H, W, 3) uint8 ndarray."""
    from PIL import Image  # noqa: WPS433

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


class PerceptionEngine:
    """Lazy loader for RAM+ Swin-L (image_size=384, vit='swin_l')."""

    _instance: "PerceptionEngine | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.ram_ckpt = SMARTWAY_RAM_PLUS_CKPT_DEFAULT
        self.device = None
        self.ram_model = None
        self.ram_transform = None
        self._loaded = False

    @classmethod
    def get(cls) -> "PerceptionEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            log.info("Loading RAM+ Swin-L from %s", self.ram_ckpt)
            import torch  # noqa: WPS433

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            # RAM+ ships in the same `ram` package as RAM but as a
            # different model factory: ``ram_plus`` vs ``ram``.
            from ram.models import ram_plus  # type: ignore
            from ram import get_transform  # type: ignore

            model = ram_plus(
                pretrained=self.ram_ckpt,
                image_size=384,
                vit="swin_l",
            )
            model.eval()
            model = model.to(self.device)
            self.ram_model = model
            self.ram_transform = get_transform(image_size=384)
            self._loaded = True
            log.info("RAM+ ready (device=%s)", self.device)

    def tag(self, rgb: np.ndarray) -> str:
        """Tag a single RGB image → space-joined tag string.

        Upstream invokes ``inference_ram(img, ram_model)`` per candidate;
        the call returns a tuple ``(tags_en, tags_chinese)`` or a single
        string depending on the RAM version. We strip the ' |' delimiter
        per the OpenNav convention so the output reads as space-tokens
        suitable for direct prompt interpolation.
        """
        self._ensure_loaded()
        from PIL import Image  # noqa: WPS433
        from ram import inference_ram  # type: ignore

        pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        image = self.ram_transform(pil).unsqueeze(0).to(self.device)
        result = inference_ram(image, self.ram_model)
        if isinstance(result, tuple):
            tags_en = result[0]
        else:
            tags_en = str(result)
        return tags_en.replace(" |", "").strip()
