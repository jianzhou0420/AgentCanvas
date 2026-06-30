"""ToolEQA system prompt — verbatim copy from upstream.

Per workspace-standalone policy, the prompt lives **inside this nodeset**
at `prompts/system_prompt.txt` (copied verbatim from the ToolEQA paper's
release at `prompt/system_prompt.txt`). Faithfulness rule R4 (no prompt
rewriting) is enforced at copy time: any future re-syncs replace this file
verbatim from the upstream snapshot.

The `<<tool_descriptions>>` and `<<authorized_imports>>` placeholders are
filled in at runtime by `transformers.agents` machinery (see
`transformers/agents/agents.py`'s `format_prompt_with_tools`).
"""

from __future__ import annotations

import os

_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "system_prompt.txt"
)


def load_system_prompt() -> str:
    """Read the local system prompt copy verbatim."""
    if not os.path.isfile(_PROMPT_PATH):
        raise FileNotFoundError(
            f"ToolEQA system_prompt.txt not found at {_PROMPT_PATH}. "
            "Restore by re-copying from upstream at "
            "third_party/zz_just_for_refer/tooleqa/prompt/system_prompt.txt."
        )
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()
