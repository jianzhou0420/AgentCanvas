"""R2R-CE instruction tokenizer — faithful replica of the offline preprocessing.

R2R-CE instruction tokens are NOT computed at runtime by upstream VLN-CE; they
ship precomputed inside the dataset json (``instruction.instruction_tokens``),
produced once when the dataset was built using habitat's ``VocabDict`` over a
fixed word-level vocabulary. AgentCanvas keeps the env emitting only RAW sensor
info (rgb/depth + raw instruction *text*), so this module re-derives the token
ids from the text — and is verified to reproduce the dataset's precomputed
``instruction_tokens`` exactly (200/200 episodes on val_unseen).

The vocab is self-contained: ``r2r_instruction_vocab.json`` (next to this
module; extracted from the dataset's ``instruction_vocab``, identical across
splits — built from train). No dependency on the env or a loaded dataset.

Recipe (mirrors ``habitat/datasets/utils.py``):
  1. ``text.lower()``
  2. strip ``,`` and ``?``                      (REQUIRED — skipping → 128/200)
  3. split on ``re.compile(r"([^\w-]+)")``, strip, drop empties
  4. word2idx, OOV → UNK_INDEX
  5. pad with PAD_INDEX to ``max_len`` (truncate if longer)
NOTE: habitat's ``tokenize`` default ``keep=("'s")`` is a STRING (not a tuple),
so upstream iterates chars ``'``/``s`` and splits on every 's'. The R2R-CE
preprocessing did NOT use that quirk (``keep=()``); replicating it drops the
match to 2/30. We deliberately do not apply ``keep``.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "r2r_instruction_vocab.json",
)
_SPLIT_REGEX = re.compile(r"([^\w-]+)")
_REMOVE = (",", "?")


@lru_cache(maxsize=1)
def _vocab() -> dict[str, Any]:
    with open(_VOCAB_PATH) as f:
        return json.load(f)


def tokenize_instruction(text: str) -> dict[str, Any]:
    """Tokenize a raw R2R-CE instruction string into the padded id sequence the
    CMA ``InstructionEncoder`` expects.

    Returns ``{"tokens": List[int] (len == max_len), "text": str}`` — the same
    shape habitat's ``InstructionSensor`` emits, so downstream
    ``extract_instruction_tokens`` consumes it unchanged.
    """
    v = _vocab()
    w2i: dict[str, int] = v["word2idx_dict"]
    unk: int = v["UNK_INDEX"]
    pad: int = v["PAD_INDEX"]
    max_len: int = v["max_len"]

    s = (text or "").lower()
    for ch in _REMOVE:
        s = s.replace(ch, "")
    words = [w.strip() for w in _SPLIT_REGEX.split(s) if w.strip()]
    ids = [w2i.get(w, unk) for w in words][:max_len]
    ids = ids + [pad] * (max_len - len(ids))
    return {"tokens": ids, "text": text}
