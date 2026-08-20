"""coding-agent/api — the API layer, abstracted out of the harnesses.

Two concerns live here and nowhere else:

- ``providers``: WHO we call — the vendor registry (key variable per vendor,
  legacy fallbacks, wire format) and the model-id → vendor resolver. Fixes
  the historical key squeeze where the DashScope key had to ride
  OPENAI_API_KEY and the OpenAI and qwen columns could not share a shell.
- ``proxy``: HOW a harness reaches a foreign vendor — an owned litellm
  gateway subprocess that translates a harness's NATIVE wire format
  (anthropic-messages for Claude Code, openai-chat for codex) to any
  litellm-routable model.

Contract with the frozen board: native pairings (sdk→Anthropic,
codex→OpenAI, mini→litellm-direct) keep their exact historical call path —
the gateway engages ONLY for cells whose extra carries
``("api_gateway", "litellm")``, and the key registry resolves to the same
variables as before unless the new canonical ones are set.
"""

from .providers import VENDORS, Vendor, resolve_key, vendor_of  # noqa: F401
from .proxy import LitellmGateway, Route  # noqa: F401
