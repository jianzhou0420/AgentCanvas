/** Group selected nodes into a composite node with auto-generated GraphIn/GraphOut. */

import type { Node, Edge } from "@xyflow/react";
import type { GraphDefinition, NodeDef, EdgeDef } from "./types";

interface GroupResult {
  compositeNode: Node;
  remainingNodes: Node[];
  remainingEdges: Edge[];
  newEdges: Edge[];
}

let groupCounter = 0;

export function groupSelectedNodes(
  selectedIds: string[],
  allNodes: Node[],
  allEdges: Edge[],
): GroupResult {
  const selectedSet = new Set(selectedIds);

  // Separate selected vs remaining nodes
  const selectedNodes = allNodes.filter((n) => selectedSet.has(n.id));
  const remainingNodes = allNodes.filter((n) => !selectedSet.has(n.id));

  // Classify edges
  const internalEdges: Edge[] = [];
  const incomingEdges: Edge[] = [];
  const outgoingEdges: Edge[] = [];
  const externalEdges: Edge[] = [];

  for (const e of allEdges) {
    const srcIn = selectedSet.has(e.source);
    const tgtIn = selectedSet.has(e.target);
    if (srcIn && tgtIn) internalEdges.push(e);
    else if (!srcIn && tgtIn) incomingEdges.push(e);
    else if (srcIn && !tgtIn) outgoingEdges.push(e);
    else externalEdges.push(e);
  }

  // Compute centroid for the composite node position
  const cx =
    selectedNodes.reduce((s, n) => s + n.position.x, 0) / selectedNodes.length;
  const cy =
    selectedNodes.reduce((s, n) => s + n.position.y, 0) / selectedNodes.length;

  // Create GraphIn nodes for incoming edges
  const graphInNodes: NodeDef[] = [];
  const graphInRewireMap = new Map<string, string>(); // old targetHandle → graphIn id
  const incomingByHandle = new Map<string, Edge>();

  for (const e of incomingEdges) {
    const handle = e.targetHandle || "input";
    if (!incomingByHandle.has(handle)) {
      incomingByHandle.set(handle, e);
      const graphInId = `graphIn_${handle}_${groupCounter}`;
      graphInNodes.push({
        id: graphInId,
        type: "graphIn",
        label: `In: ${handle}`,
        position: { x: 0, y: graphInNodes.length * 80 },
        config: { portName: handle },
      });
      graphInRewireMap.set(handle, graphInId);
    }
  }

  // Create GraphOut nodes for outgoing edges
  const graphOutNodes: NodeDef[] = [];
  const graphOutRewireMap = new Map<string, string>(); // old sourceHandle → graphOut id
  const outgoingByHandle = new Map<string, Edge>();

  for (const e of outgoingEdges) {
    const handle = e.sourceHandle || "output";
    if (!outgoingByHandle.has(handle)) {
      outgoingByHandle.set(handle, e);
      const graphOutId = `graphOut_${handle}_${groupCounter}`;
      graphOutNodes.push({
        id: graphOutId,
        type: "graphOut",
        label: `Out: ${handle}`,
        position: { x: 600, y: graphOutNodes.length * 80 },
        config: { portName: handle },
      });
      graphOutRewireMap.set(handle, graphOutId);
    }
  }

  // Build inner graph nodes (offset positions relative to min x/y)
  const minX = Math.min(...selectedNodes.map((n) => n.position.x));
  const minY = Math.min(...selectedNodes.map((n) => n.position.y));
  const innerNodes: NodeDef[] = selectedNodes.map((n) => ({
    id: n.id,
    type: n.type || "unknown",
    label: (n.data?.label as string) || n.type || "",
    position: { x: n.position.x - minX + 120, y: n.position.y - minY },
    config: Object.fromEntries(
      Object.entries(n.data || {}).filter(([k]) => k !== "label"),
    ),
  }));

  // Build inner edges: internal + graphIn→target + source→graphOut
  const innerEdges: EdgeDef[] = internalEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle || undefined,
    targetHandle: e.targetHandle || undefined,
  }));

  // Wire graphIn → original targets
  for (const e of incomingEdges) {
    const handle = e.targetHandle || "input";
    const graphInId = graphInRewireMap.get(handle);
    if (graphInId) {
      innerEdges.push({
        id: `e_pin_${graphInId}_${e.target}`,
        source: graphInId,
        target: e.target,
        sourceHandle: handle,
        targetHandle: e.targetHandle || undefined,
      });
    }
  }

  // Wire original sources → graphOut
  for (const e of outgoingEdges) {
    const handle = e.sourceHandle || "output";
    const graphOutId = graphOutRewireMap.get(handle);
    if (graphOutId) {
      innerEdges.push({
        id: `e_pout_${e.source}_${graphOutId}`,
        source: e.source,
        target: graphOutId,
        sourceHandle: e.sourceHandle || undefined,
        targetHandle: handle,
      });
    }
  }

  // Build the GraphDefinition
  const subgraph: GraphDefinition = {
    name: `Composite ${++groupCounter}`,
    description: `Grouped from ${selectedNodes.length} nodes`,
    nodes: [...graphInNodes, ...innerNodes, ...graphOutNodes],
    edges: innerEdges,
  };

  // Create the composite node
  const compositeId = `composite_${groupCounter}`;
  const compositeNode: Node = {
    id: compositeId,
    type: "compositeNode",
    position: { x: cx, y: cy },
    data: { subgraph },
  };

  // Rewire external edges to/from the composite
  const newEdges: Edge[] = [];

  for (const e of incomingEdges) {
    const handle = e.targetHandle || "input";
    newEdges.push({
      ...e,
      id: `rewire_in_${e.id}`,
      target: compositeId,
      targetHandle: handle,
    });
  }

  for (const e of outgoingEdges) {
    const handle = e.sourceHandle || "output";
    newEdges.push({
      ...e,
      id: `rewire_out_${e.id}`,
      source: compositeId,
      sourceHandle: handle,
    });
  }

  return {
    compositeNode,
    remainingNodes,
    remainingEdges: externalEdges,
    newEdges,
  };
}
