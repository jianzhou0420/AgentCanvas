/** Graph definition types — universal data model for composable node graphs.
 *
 * These types are used at every nesting depth. A NodeDef may contain a
 * subgraph (GraphDefinition), enabling recursive composition.
 *
 * Canonical location — all canvas code should import from here.
 * Renamed from InnerNode/InnerEdge/AgentLoopDefinition as part of ADR-006.
 */

export interface NodeDef {
  id: string;
  type: string;
  label: string;
  position: { x: number; y: number };
  config?: Record<string, unknown>;
  /** If present, this node is a composite containing a nested graph. */
  subgraph?: GraphDefinition;
}

export interface EdgeDef {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

/** A single named state entry inside a ContainerDef. */
export interface StateDef {
  type: "accumulator" | "lastWrite" | "counter" | "ephemeral";
  value_type: string; // "TEXT" | "ACTION" | "POINTCLOUD" | "ANY" | ...
  config?: Record<string, unknown>;
  /** When the state clears — orthogonal to the reducer. Defaults to "forever". */
  lifetime?: "step" | "episode" | "run" | "forever" | "custom";
  /** Explicit signal subscription list — only honored when lifetime === "custom". */
  reset_on?: string[];
}

/** A state container — dict of named states, visible on canvas. */
export interface ContainerDef {
  id: string;
  label: string;
  position: { x: number; y: number };
  states: Record<string, StateDef>;
}

/** Access grant: a node can read/write a container.
 *
 * Access grants are NOT wires — they carry no data, do not trigger firing.
 * They exist solely to authorise a node's read/write calls on a container.
 * Rendered as dashed violet lines on the canvas. See ADR-026.
 */
export interface AccessGrantDef {
  id: string;
  node_id: string;
  container_id: string;
}

export interface HookDef {
  event:
    | "PreNodeExecute"
    | "PostNodeExecute"
    | "GraphStart"
    | "GraphComplete"
    | "GraphError";
  command: string;
  match_node_type?: string;
  match_node_id?: string;
  timeout_ms?: number;
  enabled?: boolean;
}

export interface GraphDefinition {
  name: string;
  description: string;
  nodes: NodeDef[];
  edges: EdgeDef[];
  /** State containers — visible shared memory on canvas.
   *
   * A container with the well-known id "graph_state" plays the role of
   * the optional graph-level blackboard (ADR-026 removed the separate
   * `graph_state` field; every node that wants it now has an explicit
   * access grant to the "graph_state" container).
   */
  containers?: ContainerDef[];
  /** Node → container access grants (not wires). */
  access_grants?: AccessGrantDef[];
  /** "graph" = openable template, "node" = draggable composite archive. */
  kind?: "graph" | "node";
  /** User-defined group for organizing graph nodes (e.g. "history", "planning"). */
  group?: string;
  /** Loop-specific fields — present on agent loop graphs, absent on simple composites. */
  presetId?: string;
  /** Per-episode iteration cap. The framework's resolver chain can
   *  override this with an env-supplied value per episode. */
  step_budget?: number | null;
  /** Eval-graph flag — when true (default), the graph must declare
   *  at least one graphOut node (its config.portName becomes the metric
   *  key). Demo / playground graphs opt out. */
  eval_graph?: boolean;
  hooks?: HookDef[];
}

export interface CatalogEntry {
  type: string;
  label: string;
  icon: string;
  category: string;
  data?: Record<string, unknown>;
}
