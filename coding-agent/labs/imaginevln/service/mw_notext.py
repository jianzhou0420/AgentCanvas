"""Drop the T5-XXL text encoder from the MemoryWorld world model.

The model's prompt is a hardcoded constant -- ``"A random room tour"``
(``scripts/demo_run.py:83``) -- so T5 recomputes the same tensor on every call
while occupying 8.87 GiB of the 12.43 GiB weight footprint. Its output is
cached to disk once (1.77 MiB) and replayed here, which is numerically exact:
same tensor in, same tensor out.

Call ``install()`` BEFORE constructing ``CogVTrainer``. Three patches:

    build_model()  stub out the T5 / tokenizer loaders so nothing is read
    encode_text()  return the cached embedding, broadcast to the batch
    build_pipe()   hand the diffusers pipeline text_encoder=None (the pipe is
                   only used for prepare_extra_step_kwargs + video_processor)
"""

from __future__ import annotations

import os

import torch

# coding-agent/imaginevln/cache/prompt_embed.pt（1.8 MB，进 git——它是常量嵌入）
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "cache", "prompt_embed.pt")
CONST_PROMPT = "A random room tour"


class _NullTextEncoder(torch.nn.Module):
    """Stands in for T5 so .to()/.requires_grad_()/unwrap_model() keep working.
    Holds no parameters, so it costs nothing on either device."""

    dtype = torch.bfloat16

    def forward(self, *a, **k):  # pragma: no cover - must never be reached
        raise RuntimeError("text encoder was removed; prompt embeddings are cached")


def load_cached_embedding(path: str = DEFAULT_CACHE):
    blob = torch.load(path, map_location="cpu")
    if blob.get("prompt") != CONST_PROMPT:
        raise ValueError(f"cached embedding is for {blob.get('prompt')!r}, "
                         f"but the model's constant prompt is {CONST_PROMPT!r}")
    return blob["embeds"]


def install(cache_path: str = DEFAULT_CACHE):
    from video_sim.trainer import cogv

    embeds = load_cached_embedding(cache_path)

    original_build_model = cogv.CogVTrainer.build_model
    original_build_pipe = cogv.CogVTrainer.build_pipe

    def build_model(self):
        real_tok, real_t5 = cogv.AutoTokenizer, cogv.T5EncoderModel
        cogv.AutoTokenizer = type("_NoTok", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
        cogv.T5EncoderModel = type("_NoT5", (), {"from_pretrained": staticmethod(lambda *a, **k: _NullTextEncoder())})
        try:
            original_build_model(self)
        finally:
            cogv.AutoTokenizer, cogv.T5EncoderModel = real_tok, real_t5

    def encode_text(self, prompts):
        n = len(prompts) if not isinstance(prompts, str) else 1
        return embeds.to(device=self.accelerator.device, dtype=self.weight_dtype).expand(n, -1, -1)

    def build_pipe(self):
        text_encoder, self.text_encoder = self.text_encoder, None
        try:
            original_build_pipe(self)
        finally:
            self.text_encoder = text_encoder

    cogv.CogVTrainer.build_model = build_model
    cogv.CogVTrainer.encode_text = encode_text
    cogv.CogVTrainer.build_pipe = build_pipe
    return embeds
