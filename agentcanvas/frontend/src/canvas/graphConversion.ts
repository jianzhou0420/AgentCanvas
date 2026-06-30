/** Conversion between GraphDefinition (NodeDef/EdgeDef) and React Flow (Node/Edge).
 *
 * Used by UnifiedGraphEditor for subgraph editing, by useFlowStore for
 * template loading, and by save/load flows.
 */

import type { Node, Edge } from "@xyflow/react";
import type {
  NodeDef,
  EdgeDef,
  GraphDefinition,
  ContainerDef,
  AccessGrantDef,
} from "./types";
import { resolveInstancePorts } from "./portResolution";

/** Convert GraphDefinition nodes to React Flow nodes.
 *
 * When *schemaMap* is provided, each node receives a ``_schema`` entry
 * so that ``GenericBlockRenderer`` can render ports, colours, and
 * config fields.  This is critical for composite subgraph nodes that
 * would otherwise render as grey blocks with no ports.
 */
export function toFlowNodes(
  innerNodes: NodeDef[],
  schemaMap?: Map<string, unknown>,
): Node[] {
  return innerNodes.map((n) => {
    const rawSchema = schemaMap?.get(n.type) as
      | Record<string, unknown>
      | undefined;
    const schema = rawSchema
      ? resolveInstancePorts(
          rawSchema,
          n.config as Record<string, unknown> | undefined,
        )
      : rawSchema;
    return {
      id: n.id,
      type: n.type,
      position: n.position,
      data: {
        label: n.label,
        ...n.config,
        ...(n.subgraph ? { subgraph: n.subgraph, innerGraph: n.subgraph } : {}),
        ...(schema && !n.config?._schema ? { _schema: schema } : {}),
      },
    };
  });
}

/** Convert GraphDefinition edges to React Flow edges. */
export function toFlowEdges(innerEdges: EdgeDef[]): Edge[] {
  return innerEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle,
    targetHandle: e.targetHandle,
    animated: true,
  }));
}

/** Convert React Flow nodes to GraphDefinition nodes. */
export function fromFlowNodes(nodes: Node[]): NodeDef[] {
  return nodes.map((n) => {
    const { label, subgraph, innerGraph, _schema, ...rest } = n.data as Record<
      string,
      unknown
    >;
    const sub = (subgraph || innerGraph) as GraphDefinition | undefined;
    return {
      id: n.id,
      type: n.type || "unknown",
      label: (label as string) || "",
      position: n.position,
      config: rest,
      ...(sub ? { subgraph: sub } : {}),
    };
  });
}

/** Convert React Flow edges to GraphDefinition edges. */
export function fromFlowEdges(edges: Edge[]): EdgeDef[] {
  return edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle || undefined,
    targetHandle: e.targetHandle || undefined,
  }));
}

// ── State container conversions ──

/** Convert ContainerDefs to React Flow nodes. */
export function toFlowContainerNodes(containers: ContainerDef[]): Node[] {
  return containers.map((c) => ({
    id: c.id,
    type: "stateContainer",
    position: c.position,
    data: {
      label: c.label,
      states: c.states,
    },
  }));
}

/** Extract ContainerDefs from React Flow nodes. */
export function fromFlowContainerNodes(nodes: Node[]): ContainerDef[] {
  return nodes
    .filter((n) => n.type === "stateContainer")
    .map((n) => ({
      id: n.id,
      label: (n.data?.label as string) || "",
      position: n.position,
      states: (n.data?.states as ContainerDef["states"]) || {},
    }));
}

/** Convert AccessGrantDefs to React Flow edges (dashed violet lines). */
export function toFlowAccessGrants(grants: AccessGrantDef[]): Edge[] {
  return grants.map((ag) => ({
    id: ag.id,
    source: ag.node_id,
    target: ag.container_id,
    sourceHandle: "__state__",
    targetHandle: "state",
    type: "accessGrant",
  }));
}

/** Extract AccessGrantDefs from React Flow edges. */
export function fromFlowAccessGrants(edges: Edge[]): AccessGrantDef[] {
  return edges
    .filter((e) => e.type === "accessGrant")
    .map((e) => ({
      id: e.id,
      node_id: e.source,
      container_id: e.target,
    }));
}

// ── Full graph conversion ──

/** Convert a full GraphDefinition to React Flow nodes + edges. */
export function graphToFlow(
  graph: GraphDefinition,
  schemaMap?: Map<string, unknown>,
): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: [
      ...toFlowNodes(graph.nodes, schemaMap),
      ...toFlowContainerNodes(graph.containers || []),
    ],
    edges: [
      ...toFlowEdges(graph.edges),
      ...toFlowAccessGrants(graph.access_grants || []),
    ],
  };
}

/** Convert React Flow nodes + edges back to a GraphDefinition. */
export function flowToGraph(
  nodes: Node[],
  edges: Edge[],
  base?: Partial<GraphDefinition>,
): GraphDefinition {
  const regularNodes = nodes.filter((n) => n.type !== "stateContainer");
  const containerNodes = nodes.filter((n) => n.type === "stateContainer");
  const dataEdges = edges.filter((e) => e.type !== "accessGrant");
  const grantEdges = edges.filter((e) => e.type === "accessGrant");

  return {
    name: base?.name || "",
    description: base?.description || "",
    nodes: fromFlowNodes(regularNodes),
    edges: fromFlowEdges(dataEdges),
    containers: fromFlowContainerNodes(containerNodes),
    access_grants: fromFlowAccessGrants(grantEdges),
    step_budget: base?.step_budget ?? 500,
    presetId: base?.presetId,
  };
}
