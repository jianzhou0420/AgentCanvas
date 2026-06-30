/** CompositeCanvasNode — standalone composite node (no playback controls).
 * Registered as 'compositeNode' in the node type registry. */

import type { NodeProps } from "@xyflow/react";
import { Handle, Position } from "@xyflow/react";
import CompositeNodeView from "./CompositeNodeView";
import type { GraphDefinition } from "../../types";

export default function CompositeCanvasNode({ id, data }: NodeProps) {
  const subgraph = (data.subgraph || data.innerGraph) as
    | GraphDefinition
    | undefined;

  return (
    <div className="relative">
      <CompositeNodeView
        id={id}
        subgraph={subgraph}
        title={subgraph?.name}
        color="bg-blue-800"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="__state__"
        className="state-handle"
        style={{
          background: "#8b5cf6",
          width: 6,
          height: 6,
          border: "1.5px solid #1f2937",
        }}
      />
    </div>
  );
}
