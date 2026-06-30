/** Default graph — empty canvas.
 *
 * Agent presets (Straightforward, NavGPT-CE) are saved as JSON files in
 * workspace/graphs/ and loaded from the sidebar "Saved Graphs" section.
 */

import type { Node, Edge } from "@xyflow/react";

export interface GraphTemplate {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

export const TEMPLATES: GraphTemplate[] = [
  {
    id: "empty",
    name: "Empty Canvas",
    description: "Start from scratch — drag nodes from the sidebar",
    nodes: [],
    edges: [],
  },
];

export const defaultNodes: Node[] = [];
export const defaultEdges: Edge[] = [];
