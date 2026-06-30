/** CompositeNodeView — generic node that holds a subgraph with GraphIn/GraphOut-derived handles.
 *
 * Used by CompositeCanvasNode for reusable node groups.
 * External ports are derived from graphIn/graphOut nodes inside the subgraph.
 */

import { useState, useMemo, type ReactNode } from "react";
import { Position } from "@xyflow/react";
import { Box, ChevronDown, ChevronRight, Maximize2 } from "lucide-react";
import clsx from "clsx";
import NodeShell from "../shared/NodeShell";
import HandleDot from "../shared/HandleDot";
import { useFlowStore } from "../../useFlowStore";
import type { GraphDefinition } from "../../types";
import type { NavStatus } from "../../../types";

interface CompositeNodeViewProps {
  id: string;
  subgraph: GraphDefinition | undefined;
  title?: string;
  icon?: ReactNode;
  color?: string;
  status?: NavStatus | "inactive";
  children?: ReactNode;
  className?: string;
}

/** Derive input ports from graphIn nodes in the subgraph. */
function deriveInputPorts(graph: GraphDefinition | undefined): string[] {
  if (!graph?.nodes) return [];
  return graph.nodes
    .filter((n) => n.type === "graphIn")
    .map((n) => (n.config?.portName as string) || "input");
}

/** Derive output ports from graphOut nodes in the subgraph. */
function deriveOutputPorts(graph: GraphDefinition | undefined): string[] {
  if (!graph?.nodes) return [];
  return graph.nodes
    .filter((n) => n.type === "graphOut")
    .map((n) => (n.config?.portName as string) || "output");
}

export { deriveInputPorts, deriveOutputPorts };

export default function CompositeNodeView({
  id,
  subgraph,
  title,
  icon,
  color = "bg-blue-800",
  status = "inactive",
  children,
  className,
}: CompositeNodeViewProps) {
  const [expanded, setExpanded] = useState(false);
  const openSubgraph = useFlowStore((s) => s.openSubgraph);

  const name = title || subgraph?.name || "Composite";
  const inputPorts = useMemo(() => deriveInputPorts(subgraph), [subgraph]);
  const outputPorts = useMemo(() => deriveOutputPorts(subgraph), [subgraph]);

  return (
    <NodeShell
      title={name}
      icon={icon || <Box size={12} />}
      color={color}
      status={status}
      className={clsx(
        "!min-w-[250px]",
        expanded && "!max-w-[520px]",
        className,
      )}
    >
      {/* Input handles from graphIn nodes */}
      {inputPorts.map((port, i) => {
        const topPct = Math.round(
          15 + (i * 70) / Math.max(inputPorts.length - 1, 1),
        );
        return (
          <div key={`in-${port}`}>
            <HandleDot
              type="target"
              position={Position.Left}
              handleType="stream"
              id={port}
              style={{ top: `${topPct}%` }}
            />
            <div
              className="absolute text-[8px] text-blue-400"
              style={{
                left: -2,
                top: `${topPct - 2}%`,
                transform: "translateX(-100%)",
              }}
            >
              {port}
            </div>
          </div>
        );
      })}

      {/* Subgraph info */}
      {subgraph && (
        <div className="text-[10px] text-gray-500">
          {subgraph.description || ""}
        </div>
      )}

      {/* Preview + edit */}
      {subgraph && subgraph.nodes.length > 0 && (
        <>
          <div className="flex gap-1">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex flex-1 items-center gap-1 rounded px-1 py-0.5 text-[10px] text-gray-500 hover:bg-gray-800 hover:text-gray-300"
            >
              {expanded ? (
                <ChevronDown size={10} />
              ) : (
                <ChevronRight size={10} />
              )}
              {expanded ? "Hide" : "Preview"}
              <span className="ml-auto text-gray-600">
                {subgraph.nodes.length} nodes
              </span>
            </button>
            <button
              onClick={() => openSubgraph(id)}
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-purple-400 hover:bg-purple-900/30 hover:text-purple-300"
              title="Edit subgraph"
            >
              <Maximize2 size={10} /> Edit
            </button>
          </div>

          {expanded && (
            <div className="mt-1 rounded border border-gray-700 bg-gray-900/50 p-2 text-[10px] text-gray-500">
              {subgraph.nodes
                .map((n) => n.type)
                .filter((v, i, a) => a.indexOf(v) === i)
                .join(", ")}
            </div>
          )}
        </>
      )}

      {/* Slot for additional controls */}
      {children}

      {/* Output handles from graphOut nodes */}
      {outputPorts.map((port, i) => {
        const topPct = Math.round(
          15 + (i * 70) / Math.max(outputPorts.length - 1, 1),
        );
        return (
          <div key={`out-${port}`}>
            <HandleDot
              type="source"
              position={Position.Right}
              handleType="stream"
              id={port}
              style={{ top: `${topPct}%` }}
            />
            <div
              className="absolute text-[8px] text-orange-400"
              style={{
                right: -2,
                top: `${topPct - 2}%`,
                transform: "translateX(100%)",
              }}
            >
              {port}
            </div>
          </div>
        );
      })}
    </NodeShell>
  );
}
