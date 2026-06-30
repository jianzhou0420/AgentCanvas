"""Batched inference rendezvous tier — ADR-028 PC-2.

Sits inside an :class:`AutoServerApp` subprocess. K worker LoopRunners (in
the canvas/eval main process) POST concurrent ``/call/{fn}`` requests for
the same batched node; the server collects them per ``(function_name,
config_hash)``, calls the underlying handler **once** with all K samples
stacked, and scatters K result slices back to K awaiting HTTP responses.

Why server-side instead of client-side rendezvous:
    * The K workers run in **separate** ``LoopRunner`` instances inside one
      process; their ``policy_cma__forward`` calls all proxy to the same
      policy subprocess. The natural seam to batch them is **server-side**,
      not client-side, because that's where the K calls converge.
    * Keeping the rendezvous in-process (within the policy subprocess)
      avoids inventing a separate inference-server subprocess. One
      ``AutoServerApp`` per nodeset, one subprocess, one ``/health`` —
      ADR-028 rule (6e).

Why batch key is ``(function_name, config_hash)`` and not ``(node.id, ...)``
as the ADR text suggests:
    * The server doesn't see graph ``node.id`` — that's a canvas concept.
    * ``config_hash`` already prevents cross-batching of two policy nodes
      pointing at different checkpoints, which was the only failure mode
      ``node.id`` was guarding against. Two nodes with **identical** config
      that get cross-batched are mathematically equivalent to two
      independent calls — sharing one forward pass is the correct outcome.

Pure-functional contract: the server holds **no per-worker state**. RNN
hidden states must travel as explicit input/output ports
(``hidden_in``/``hidden_out``) so they ride the wire and stay with the
caller — see ADR-028 rule (6c).

Flush rule (ADR-028 rule 6d): per-key flush fires ``flush_timeout_ms``
after the **last** arrival (restart-on-each-submit timer). A late caller
joins the in-flight batch instead of forcing the partial batch out early,
which keeps the rendezvous correct even when workers arrive at uneven
times.

Single-in-flight: a server-level ``_inference_lock`` serialises handler
calls across all keys. One ``AutoServerApp`` owns one device, so peak
device memory is bounded by a single batch regardless of how many keys
are active. The lock is held only around the underlying handler call,
not around queue manipulation, so submits keep arriving and accumulating
while a batch is on the device.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Tuple

log = logging.getLogger("agentcanvas.batched-inference")

# Batch key shape: (function_name, config_hash). See module docstring.
# Typing.Tuple form (not ``tuple[...]``) so this module imports under the
# Python 3.8 vlnce env that hosts the habitat / policy_cma server-mode
# subprocesses. ``from __future__ import annotations`` doesn't cover
# module-level type aliases (they're evaluated at import time).
BatchKey = Tuple[str, str]

# Marker keys on the batched-handler boundary. The server passes
# ``{_samples: [inputs_dict, ...]}`` into ``tool.forward`` and expects
# ``{_outputs: [outputs_dict, ...]}`` back. The node owns per-port stacking
# semantics (some ports stack as torch tensors, others as lists of dicts).
SAMPLES_KEY = "_samples"
OUTPUTS_KEY = "_outputs"


def config_hash(config: dict | None) -> str:
    """Stable 8-char sha-1 of a JSON-serialisable config dict.

    Order-insensitive via ``sort_keys`` so two callers with equivalent
    config (any key order) share a batch queue.
    """
    if not config:
        return "00000000"
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


@dataclass
class _PendingCall:
    inputs: dict
    future: asyncio.Future


# Underlying handler signature: (samples, config) -> outputs (one per sample).
BatchedHandler = Callable[[List[dict], dict], Awaitable[List[dict]]]


class _BatchQueue:
    """One queue per batch key. Owns its own restart-on-submit flush timer."""

    def __init__(
        self,
        key: BatchKey,
        handler: BatchedHandler,
        flush_timeout_ms: int,
        inference_lock: asyncio.Lock,
    ) -> None:
        self.key = key
        self.handler = handler
        self.flush_timeout_ms = flush_timeout_ms
        self._pending: list[_PendingCall] = []
        self._latest_config: dict = {}
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._inference_lock = inference_lock

    async def submit(self, inputs: dict, config: dict) -> dict:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        async with self._lock:
            self._pending.append(_PendingCall(inputs=inputs, future=future))
            # All callers in this queue share config_hash → semantically
            # equivalent configs. Storing the latest is fine.
            self._latest_config = config
            if self._flush_task is not None and not self._flush_task.done():
                self._flush_task.cancel()
            self._flush_task = asyncio.create_task(self._delayed_flush())
        return await future

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self.flush_timeout_ms / 1000.0)
        except asyncio.CancelledError:
            return
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._pending:
                self._flush_task = None
                return
            batch = self._pending
            self._pending = []
            config = self._latest_config
            self._flush_task = None

        log.debug("flush key=%s batch_size=%d", self.key, len(batch))
        sample_inputs = [c.inputs for c in batch]
        try:
            async with self._inference_lock:
                outputs = await self.handler(sample_inputs, config)
        except Exception as e:
            log.exception("Batched handler raised for key=%s", self.key)
            for call in batch:
                if not call.future.done():
                    call.future.set_exception(e)
            return

        if not isinstance(outputs, list) or len(outputs) != len(batch):
            err = RuntimeError(
                f"Batched handler for {self.key} returned "
                f"{type(outputs).__name__} of len "
                f"{len(outputs) if hasattr(outputs, '__len__') else '?'}; "
                f"expected list of len {len(batch)}"
            )
            for call in batch:
                if not call.future.done():
                    call.future.set_exception(err)
            return

        # ``strict=`` kwarg is Py 3.10+; this module also imports under the
        # vlnce env (3.8), so we can't pass strict=True nor strict=False. The
        # length-equality check above already enforces the invariant ruff
        # B905 wants — silence the lint here, not project-wide.
        for call, out in zip(batch, outputs):  # noqa: B905
            if not call.future.done():
                call.future.set_result(out)


class BatchedInferenceServer:
    """In-process rendezvous tier — one per :class:`AutoServerApp`.

    Lazily creates one :class:`_BatchQueue` per ``BatchKey`` on first
    submission. Handlers are registered eagerly at startup so that the
    first submission of a key has somewhere to land.
    """

    def __init__(self, flush_timeout_ms: int = 50) -> None:
        self.flush_timeout_ms = flush_timeout_ms
        self._queues: dict[BatchKey, _BatchQueue] = {}
        self._handlers: dict[str, BatchedHandler] = {}  # keyed by function_name
        self._lock = asyncio.Lock()
        # Server-level single-in-flight: only one batched handler runs on the
        # underlying device at a time, across all keys. Bounds peak GPU memory
        # to one batch; cross-key serialization is harmless because one server
        # owns one device.
        self._inference_lock = asyncio.Lock()

    def register(self, function_name: str, handler: BatchedHandler) -> None:
        """Bind a function name to its batched handler. Idempotent."""
        self._handlers[function_name] = handler

    async def submit(
        self,
        function_name: str,
        inputs: dict,
        config: dict,
    ) -> dict:
        """Submit one sample, await the per-flush slice."""
        key: BatchKey = (function_name, config_hash(config))
        async with self._lock:
            queue = self._queues.get(key)
            if queue is None:
                handler = self._handlers.get(function_name)
                if handler is None:
                    raise KeyError(
                        f"BatchedInferenceServer: no handler registered for "
                        f"function {function_name!r}"
                    )
                queue = _BatchQueue(key, handler, self.flush_timeout_ms, self._inference_lock)
                self._queues[key] = queue
        return await queue.submit(inputs, config)

    async def shutdown(self) -> None:
        """Cancel pending flush timers and drop queues. Safe to call twice."""
        for q in list(self._queues.values()):
            if q._flush_task is not None and not q._flush_task.done():
                q._flush_task.cancel()
        self._queues.clear()
