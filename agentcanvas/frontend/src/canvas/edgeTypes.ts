/** Custom edge types for the canvas.
 *
 * - Default edges: animated data-flow wires (React Flow built-in)
 * - accessGrant: dashed access-grant lines between nodes and state containers
 *   (not wires — they grant read/write access; see ADR-026)
 */

import AccessGrantEdge from "./edges/AccessGrantEdge";
import RoutedEdge from "./edges/RoutedEdge";

export const customEdgeTypes = {
  accessGrant: AccessGrantEdge,
  routed: RoutedEdge,
};

// Edge colors by handle type (used in defaultGraph edge styling)
export const EDGE_COLORS: Record<string, string> = {
  env: "#22c55e",
  task: "#3b82f6",
  model: "#eab308",
  tool: "#06b6d4",
  stream: "#f97316",
};
