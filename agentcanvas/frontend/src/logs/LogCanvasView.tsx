/** In-Canvas log replay — renders the executed graph with step-by-step navigation.
 *
 * Reuses the same node types (proxiedNodeTypes) and graph conversion
 * (graphToFlow) as the real canvas editor, so the graph looks identical.
 * Fetches node schemas from the backend for proper port/color rendering.
 * Applies status highlighting (firing/completed/error) via React Flow
 * node style overrides.
 */

import {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  Suspense,
} from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import clsx from "clsx";
import { logApi } from "./logApi";
import { resolveRenderer, ErrorRenderer } from "./renderers/registry";
import type { LogEntry } from "./types";
import { LogContext } from "./LogContext";
import { proxiedNodeTypes } from "../canvas/unifiedNodeTypes";
import {
  toFlowNodes,
  toFlowEdges,
  toFlowContainerNodes,
  toFlowAccessGrants,
} from "../canvas/graphConversion";
import type {
  NodeDef,
  EdgeDef,
  ContainerDef,
  AccessGrantDef,
} from "../canvas/types";
import { api } from "../api";

// ── Types ──

interface GraphDef {
  nodes: NodeDef[];
  edges: EdgeDef[];
  containers?: ContainerDef[];
  access_grants?: AccessGrantDef[];
}

interface Props {
  executionId: string;
}

// ── Helpers ──

function formatDuration(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Status highlight styles applied on top of each node. */
const STATUS_STYLES: Record<string, React.CSSProperties> = {
  firing: {
    outline: "2px solid #3b82f6",
    outlineOffset: "2px",
    boxShadow: "0 0 16px rgba(59,130,246,0.5)",
    borderRadius: "8px",
    transition: "all 0.2s ease",
  },
  completed: {
    outline: "1px solid #22c55e40",
    outlineOffset: "1px",
    borderRadius: "8px",
    transition: "all 0.2s ease",
  },
  error: {
    outline: "2px solid #ef4444",
    outlineOffset: "2px",
    borderRadius: "8px",
    transition: "all 0.2s ease",
  },
};

// ── Rendered values (reused from LogEntryList pattern) ──

function RenderedValues({
  data,
  portWireTypes,
}: {
  data: Record<string, unknown>;
  portWireTypes: Record<string, string>;
}) {
  return (
    <div className="space-y-1.5">
      {Object.entries(data).map(([key, val]) => {
        const wireType = portWireTypes[key];
        const Renderer = resolveRenderer(val, wireType);
        return (
          <div key={key}>
            <div className="flex items-center gap-1 mb-0.5">
              <span className="text-[10px] text-gray-500">{key}</span>
              {wireType && (
                <span className="text-[9px] text-gray-600 bg-gray-800 px-1 rounded">
                  {wireType}
                </span>
              )}
            </div>
            <Suspense
              fallback={
                <span className="text-gray-600 text-[10px]">loading...</span>
              }
            >
              <Renderer value={val} label={key} />
            </Suspense>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Component ──

export default function LogCanvasView({ executionId }: Props) {
  const [graph, setGraph] = useState<GraphDef | null>(null);
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [schemas, setSchemas] = useState<Map<string, unknown> | undefined>(
    undefined,
  );
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [panelHeight, setPanelHeight] = useState(208);
  const dragRef = useRef<{ startY: number; startH: number } | null>(null);

  // Fetch graph + entries + schemas in parallel
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setGraphError(null);
    setCurrentIdx(0);
    setPlaying(false);

    Promise.all([
      logApi.getGraph(executionId).catch(() => null),
      logApi.getEntries(executionId, { limit: 5000 }),
      api.getNodeSchemas().catch(() => []),
    ]).then(([graphData, logData, schemaData]) => {
      if (cancelled) return;
      if (!graphData) {
        setGraphError(
          "No graph.json saved for this execution. Run the graph again to enable canvas replay.",
        );
      } else {
        setGraph(graphData as unknown as GraphDef);
      }
      setEntries(logData.entries);
      if (Array.isArray(schemaData) && schemaData.length > 0) {
        const sMap = new Map(
          schemaData.map((s: Record<string, unknown>) => [s.type as string, s]),
        );
        setSchemas(sMap);
      }
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [executionId]);

  // Auto-play
  useEffect(() => {
    if (!playing || entries.length === 0) return;
    const timer = setInterval(() => {
      setCurrentIdx((prev) => {
        if (prev >= entries.length - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 800);
    return () => clearInterval(timer);
  }, [playing, entries.length]);

  // Current entry and cumulative status sets
  const currentEntry = entries[currentIdx] ?? null;

  const { firedNodeIds, errorNodeIds } = useMemo(() => {
    const fired = new Set<string>();
    const errors = new Set<string>();
    for (let i = 0; i <= currentIdx && i < entries.length; i++) {
      fired.add(entries[i].node_id);
      if (entries[i].error) errors.add(entries[i].node_id);
    }
    return { firedNodeIds: fired, errorNodeIds: errors };
  }, [entries, currentIdx]);

  // Build React Flow nodes/edges using the real canvas conversion
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!graph) return { flowNodes: [], flowEdges: [] };

    // Convert using the same functions as the canvas editor
    const nodes: Node[] = [
      ...toFlowNodes(graph.nodes, schemas),
      ...toFlowContainerNodes(graph.containers || []),
    ];
    const edges: Edge[] = [
      ...toFlowEdges(graph.edges),
      ...toFlowAccessGrants(graph.access_grants || []),
    ];

    // Apply status highlighting as style overrides
    const firingId = currentEntry?.node_id ?? null;
    for (const node of nodes) {
      node.draggable = false;
      node.connectable = false;
      node.selectable = false;

      if (node.id === firingId) {
        node.style = { ...node.style, ...STATUS_STYLES.firing };
      } else if (errorNodeIds.has(node.id)) {
        node.style = { ...node.style, ...STATUS_STYLES.error };
      } else if (firedNodeIds.has(node.id)) {
        node.style = { ...node.style, ...STATUS_STYLES.completed };
      }
    }

    return { flowNodes: nodes, flowEdges: edges };
  }, [graph, schemas, currentEntry, firedNodeIds, errorNodeIds]);

  // Step controls
  const goFirst = useCallback(() => {
    setCurrentIdx(0);
    setPlaying(false);
  }, []);
  const goPrev = useCallback(() => {
    setCurrentIdx((i) => Math.max(0, i - 1));
    setPlaying(false);
  }, []);
  const goNext = useCallback(() => {
    setCurrentIdx((i) => Math.min(entries.length - 1, i + 1));
    setPlaying(false);
  }, [entries.length]);
  const goLast = useCallback(() => {
    setCurrentIdx(entries.length - 1);
    setPlaying(false);
  }, [entries.length]);
  const togglePlay = useCallback(() => {
    if (currentIdx >= entries.length - 1) setCurrentIdx(0);
    setPlaying((p) => !p);
  }, [currentIdx, entries.length]);

  // Panel resize via drag handle
  const onResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragRef.current = { startY: e.clientY, startH: panelHeight };
      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const delta = dragRef.current.startY - ev.clientY;
        setPanelHeight(
          Math.max(80, Math.min(600, dragRef.current.startH + delta)),
        );
      };
      const onUp = () => {
        dragRef.current = null;
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [panelHeight],
  );

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600 text-xs">
        Loading canvas replay...
      </div>
    );
  }

  if (graphError) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 text-xs px-8 text-center">
        {graphError}
      </div>
    );
  }

  return (
    <LogContext.Provider value={{ executionId, viewMode: "canvas" }}>
      <div className="flex flex-col flex-1 min-h-0">
        {/* React Flow canvas — same node types as the editor */}
        <div className="flex-1 min-h-0 bg-gray-950">
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={proxiedNodeTypes as unknown as NodeTypes}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            panOnDrag
            zoomOnScroll
            fitView
            fitViewOptions={{ padding: 0.15 }}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#1f2937" gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        {/* Step controls */}
        <div className="flex items-center gap-2 px-3 py-1.5 border-t border-gray-800 bg-gray-900 shrink-0">
          <button
            onClick={goFirst}
            className="text-gray-400 hover:text-gray-200 text-xs px-1"
            title="First"
          >
            |&#9664;
          </button>
          <button
            onClick={goPrev}
            className="text-gray-400 hover:text-gray-200 text-xs px-1"
            title="Previous"
          >
            &#9664;
          </button>
          <button
            onClick={togglePlay}
            className={clsx(
              "px-2 py-0.5 text-xs rounded border",
              playing
                ? "border-amber-600 bg-amber-600/20 text-amber-300"
                : "border-blue-600 bg-blue-600/20 text-blue-300",
            )}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            onClick={goNext}
            className="text-gray-400 hover:text-gray-200 text-xs px-1"
            title="Next"
          >
            &#9654;
          </button>
          <button
            onClick={goLast}
            className="text-gray-400 hover:text-gray-200 text-xs px-1"
            title="Last"
          >
            &#9654;|
          </button>

          {currentEntry && (
            <div className="flex items-center gap-2 ml-2 text-[11px]">
              <span className="text-gray-400">
                <span className="text-gray-200 font-medium">
                  {currentIdx + 1}
                </span>
                /{entries.length}
              </span>
              <span className="text-gray-500">Step #{currentEntry.step}</span>
              <span
                className={clsx(
                  "font-medium",
                  currentEntry.error ? "text-red-400" : "text-gray-200",
                )}
              >
                {currentEntry.node_label}
              </span>
              <span className="text-gray-600 text-[10px]">
                {currentEntry.node_type}
              </span>
              <span className="text-gray-500 text-[10px]">
                {formatDuration(currentEntry.duration_ms)}
              </span>
            </div>
          )}
        </div>

        {/* Resize handle */}
        <div
          onMouseDown={onResizeStart}
          className="h-1.5 shrink-0 cursor-row-resize bg-gray-800 hover:bg-blue-600/40 transition-colors border-t border-gray-700"
        />

        {/* Current node detail panel — resizable */}
        <div
          style={{ height: panelHeight }}
          className="overflow-y-auto bg-gray-900/70 px-3 py-2 shrink-0"
        >
          {currentEntry ? (
            <div className="flex gap-6">
              {/* Outputs */}
              {currentEntry.outputs &&
                Object.keys(currentEntry.outputs).length > 0 && (
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-gray-500 font-semibold mb-1">
                      OUTPUTS
                    </div>
                    <RenderedValues
                      data={currentEntry.outputs}
                      portWireTypes={currentEntry.port_wire_types ?? {}}
                    />
                  </div>
                )}

              {/* Inputs */}
              {currentEntry.inputs &&
                Object.keys(currentEntry.inputs).length > 0 && (
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-gray-500 font-semibold mb-1">
                      INPUTS
                    </div>
                    <RenderedValues
                      data={currentEntry.inputs}
                      portWireTypes={currentEntry.port_wire_types ?? {}}
                    />
                  </div>
                )}

              {/* Inner log */}
              {currentEntry.inner_log && currentEntry.inner_log.length > 0 && (
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-amber-500 font-semibold mb-1">
                    NODE LOG
                  </div>
                  <div className="space-y-1">
                    {currentEntry.inner_log.map((item, i) => {
                      const Renderer = resolveRenderer(item.value);
                      return (
                        <div key={i}>
                          <span className="text-[10px] text-gray-500">
                            {item.key}:{" "}
                          </span>
                          <Suspense
                            fallback={
                              <span className="text-gray-600 text-[10px]">
                                ...
                              </span>
                            }
                          >
                            <Renderer value={item.value} label={item.key} />
                          </Suspense>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Error */}
              {currentEntry.error && (
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-red-400 font-semibold mb-1">
                    ERROR
                  </div>
                  <Suspense
                    fallback={
                      <span className="text-gray-600 text-[10px]">
                        loading...
                      </span>
                    }
                  >
                    <ErrorRenderer value={currentEntry.error} />
                  </Suspense>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-600 text-xs">
              No entry selected
            </div>
          )}
        </div>
      </div>
    </LogContext.Provider>
  );
}
