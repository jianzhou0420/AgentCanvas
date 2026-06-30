/**
 * Resolve instance-level ports from config.ports, overriding class-level schema ports.
 * Type-agnostic: works for any node type with config.ports.
 *
 * Behaviour is driven by `schema.ports_mode` (set by the backend):
 * - "sink":   config.ports → input_ports only, output_ports = [] (e.g. iterOut)
 * - "source": config.ports → output_ports only, input_ports = [] (e.g. iterIn)
 * - "input":  config.ports → input_ports only, output_ports unchanged (e.g. llmCall)
 * - "mirror": config.ports → mirrored to BOTH input_ports and output_ports
 *
 * iterIn uses the unified per-port persist schema: one ``ports`` list
 * (same shape as other configurable-port nodes), each entry producing one
 * unprefixed output handle. The per-entry ``persist: boolean`` flag is
 * surfaced on the PortDef so the canvas renderer can expose it if needed.
 *
 * Returns a new schema object (never mutates the original).
 * Returns schema unchanged if no port config is found.
 */

interface PortConfig {
  name: string;
  wire_type: string;
  persist?: boolean;
  /** iterIn synthesised ports only — every handle is always-prefixed so
   *  the init and iterOut writers have disjoint namespaces. */
  origin?: "init" | "iterOut";
  /** iterIn only: original name on the writer node (without the prefix). */
  writer_name?: string;
}

interface PortDef {
  name: string;
  wire_type: string;
  description?: string;
  optional?: boolean;
  persist?: boolean;
  origin?: "init" | "iterOut";
  writer_name?: string;
}

function toPortDefs(ports: unknown): PortDef[] {
  if (!Array.isArray(ports)) return [];
  return (ports as PortConfig[]).map((p) => ({
    name: p.name,
    wire_type: p.wire_type,
    description: "",
    optional: true,
    ...(p.persist !== undefined ? { persist: p.persist } : {}),
    ...(p.origin !== undefined ? { origin: p.origin } : {}),
    ...(p.writer_name !== undefined ? { writer_name: p.writer_name } : {}),
  }));
}

export function resolveInstancePorts(
  schema: Record<string, unknown> | undefined | null,
  config: Record<string, unknown> | undefined | null,
): Record<string, unknown> | undefined | null {
  if (!schema) return schema;

  const portsMode = (schema.ports_mode as string) || "mirror";

  const ports = config?.ports;

  // iterOut — two-sided pivot (mirror of backend IterOutNode._resolve_ports):
  // inputs = class-level ``stop`` (BOOL halt signal) + the loop-carry ports;
  // outputs = one ``final_<name>`` per loop-carry port + constant
  // ``final_stop``. The final side emits once, at scope termination.
  if ((schema.type as string) === "iterOut") {
    const carry: PortDef[] = toPortDefs(Array.isArray(ports) ? ports : []);
    const stopPort: PortDef = {
      name: "stop",
      wire_type: "BOOL",
      description: "Halt signal — truthy ends this loop",
      optional: true,
    };
    const finals: PortDef[] = carry.map((p) => ({
      name: `final_${p.name}`,
      wire_type: p.wire_type,
      description: "Terminal-iteration value (emits once at scope termination)",
      optional: true,
    }));
    finals.push({
      name: "final_stop",
      wire_type: "BOOL",
      description: "True once at scope termination — after-loop trigger",
      optional: true,
    });
    return {
      ...schema,
      input_ports: [stopPort, ...carry],
      output_ports: finals,
    };
  }

  if (!Array.isArray(ports) || ports.length === 0) return schema;

  const portDefs: PortDef[] = toPortDefs(ports);

  if (portsMode === "sink") {
    // Sink node: ports go to inputs only, outputs stay empty
    return { ...schema, input_ports: portDefs, output_ports: [] };
  }
  if (portsMode === "source") {
    // Source node (iterIn, imageViewer): ports go to outputs only
    return { ...schema, input_ports: [], output_ports: portDefs };
  }
  if (portsMode === "input") {
    // Input-only: ports override inputs, outputs unchanged from schema
    return { ...schema, input_ports: portDefs };
  }

  // 'mirror' (default): ports mirror to both sides
  return { ...schema, input_ports: portDefs, output_ports: portDefs };
}

// ---------------------------------------------------------------------------
// iterIn port synthesis (frontend mirror of backend _synthesize_iterin_ports)
// ---------------------------------------------------------------------------
//
// Computes iterIn.data.ports from its own initPorts + paired iterOut +
// direct canvas edges. Kept aligned with the Python helper so live in-canvas
// edits reflect immediately without a round-trip through the backend.

interface AnyNode {
  id: string;
  type?: string;
  data: Record<string, unknown>;
}
interface AnyEdge {
  target: string;
  targetHandle?: string | null;
}

export function synthesizeIterInPortsForId(
  iterInId: string,
  nodes: AnyNode[],
  edges: AnyEdge[],
): PortConfig[] {
  const iterIn = nodes.find((n) => n.id === iterInId && n.type === "iterIn");
  if (!iterIn) return [];
  const pairedOutId = (iterIn.data as Record<string, unknown>).pairedWith as
    | string
    | undefined;

  const out: PortConfig[] = [];
  const seen = new Set<string>();

  const emit = (
    p: { name?: string; wire_type?: string; persist?: boolean },
    origin: "init" | "iterOut",
    prefix: string,
  ) => {
    if (!p.name) return;
    const handle = `${prefix}_${p.name}`;
    if (seen.has(handle)) return;
    seen.add(handle);
    // Default persist: iterOut-origin ports persist by default (loop-carried);
    // init-origin ports are one-shot (Step 0 only) by default.
    const defaultPersist = origin === "iterOut";
    out.push({
      name: handle,
      wire_type: p.wire_type || "ANY",
      persist: Boolean(p.persist ?? defaultPersist),
      origin,
      writer_name: p.name,
    });
  };

  // iterIn's own authored init ports (two-sided model — the left/input
  // side). Prefix "init_".
  const ownInitPorts =
    ((iterIn.data as Record<string, unknown>).initPorts as
      | PortConfig[]
      | undefined) || [];
  for (const p of ownInitPorts) emit(p, "init", "init");

  // Paired iterOut (prefix "iterout_").
  if (pairedOutId) {
    const iterOutNode = nodes.find(
      (n) => n.id === pairedOutId && n.type === "iterOut",
    );
    if (iterOutNode) {
      const ports =
        ((iterOutNode.data as Record<string, unknown>).ports as
          | PortConfig[]
          | undefined) || [];
      for (const p of ports) emit(p, "iterOut", "iterout");
    }
  }
  // Direct canvas edges targeting iterIn — keep their targetHandle as-is
  // (no prefix; these are author-chosen canvas handle names).
  for (const e of edges) {
    if (e.target !== iterInId || !e.targetHandle) continue;
    if (seen.has(e.targetHandle)) continue;
    seen.add(e.targetHandle);
    out.push({
      name: e.targetHandle,
      wire_type: "ANY",
      persist: false,
      origin: "init",
      writer_name: e.targetHandle,
    });
  }
  return out;
}
