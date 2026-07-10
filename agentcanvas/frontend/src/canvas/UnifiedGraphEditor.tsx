/** Unified graph editor — works at root canvas and inside composites.
 *
 * Two modes:
 * - root: reads/writes nodes+edges from useFlowStore (Zustand)
 * - subgraph: local state initialized from innerGraph, with save/reset
 *
 * Both modes use the same unifiedNodeTypes and share drag-drop logic.
 */

import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type ReactFlowInstance,
} from "@xyflow/react";
import { Save, RotateCcw } from "lucide-react";

import type { GraphDefinition } from "./types";
import {
  COLOR_MAP,
  CATEGORY_STYLES,
  DEFAULT_STYLE,
} from "./nodes/agentloop/inner/layouts/layoutUtils";
import {
  proxiedNodeTypes,
  OUTPUT_NODE_TYPES,
  ANNOTATION_NODE_TYPES,
  isOutputNode,
} from "./unifiedNodeTypes";
import { customEdgeTypes } from "./edgeTypes";
import {
  toFlowNodes,
  toFlowEdges,
  fromFlowNodes,
  fromFlowEdges,
} from "./graphConversion";
import { useFlowStore, genNodeId } from "./useFlowStore";

/* ── Props ── */

interface UnifiedGraphEditorProps {
  mode: "root" | "subgraph";
  /** Only used when mode === 'subgraph'. */
  innerGraph?: GraphDefinition;
  /** Node schemas from backend — used to enrich subgraph nodes with _schema. */
  nodeSchemas?: Array<{ type: string; [k: string]: unknown }> | null;
  onSave?: (updated: GraphDefinition) => void;
  onClose?: () => void;
}

export default function UnifiedGraphEditor({
  mode,
  innerGraph,
  onSave,
  nodeSchemas,
}: UnifiedGraphEditorProps) {
  // ── Root mode: state from Zustand store ──
  const storeNodes = useFlowStore((s) => s.nodes);
  const storeEdges = useFlowStore((s) => s.edges);
  const storeOnNodesChange = useFlowStore((s) => s.onNodesChange);
  const storeOnEdgesChange = useFlowStore((s) => s.onEdgesChange);
  const storeOnConnect = useFlowStore((s) => s.onConnect);
  const storeAddNode = useFlowStore((s) => s.addNode);
  const setSelectedNodeId = useFlowStore((s) => s.setSelectedNodeId);

  // ── Subgraph mode: local state ──
  const schemaMap = useMemo(() => {
    if (!nodeSchemas || nodeSchemas.length === 0) return undefined;
    return new Map(nodeSchemas.map((s) => [s.type, s]));
  }, [nodeSchemas]);

  const initialNodes = useMemo(
    () =>
      mode === "subgraph" && innerGraph
        ? toFlowNodes(innerGraph.nodes, schemaMap)
        : [],
    [mode, innerGraph, schemaMap],
  );
  const initialEdges = useMemo(
    () =>
      mode === "subgraph" && innerGraph ? toFlowEdges(innerGraph.edges) : [],
    [mode, innerGraph],
  );

  const [localNodes, setLocalNodes] = useState<Node[]>(initialNodes);
  const [localEdges, setLocalEdges] = useState<Edge[]>(initialEdges);
  const [dirty, setDirty] = useState(false);
  const rfInstance = useRef<ReactFlowInstance | null>(null);

  // Register subgraph node updater so PropertiesPanel can edit subgraph nodes
  const setSubgraphNodeUpdater = useFlowStore((s) => s.setSubgraphNodeUpdater);
  const setVisibleNodes = useFlowStore((s) => s.setVisibleNodes);
  useEffect(() => {
    if (mode === "subgraph") {
      setSubgraphNodeUpdater(
        (nodeId: string, patch: Record<string, unknown>) => {
          setLocalNodes((nds) =>
            nds.map((n) =>
              n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n,
            ),
          );
          setDirty(true);
        },
      );
      return () => setSubgraphNodeUpdater(null);
    }
  }, [mode, setSubgraphNodeUpdater]);

  // ── Access-grant visibility toggle (state containers + dashed lines) ──
  const showAccessGrants = useFlowStore((s) => s.showStateEdges);
  // ── Viewer visibility toggle (viewer nodes + incident data edges) ──
  const showViewers = useFlowStore((s) => s.showViewers);
  // ── Annotation visibility toggle (note + future sticker/link types) ──
  const showAnnotations = useFlowStore((s) => s.showAnnotations);

  // ── Unified accessors — delegate to store or local state ──
  const allNodes = mode === "root" ? storeNodes : localNodes;

  // Sync visible nodes to store so PropertiesPanel can find them
  useEffect(() => {
    setVisibleNodes(allNodes);
  }, [allNodes, setVisibleNodes]);
  const allEdges = mode === "root" ? storeEdges : localEdges;

  // Filter state containers + access grants and viewer edges based on toggles (hidden by default).
  // Viewer NODES always stay visible — only the edges touching them are hidden.
  const isViewerNode = useCallback(
    (n: Node) =>
      OUTPUT_NODE_TYPES.has(n.type ?? "") ||
      isOutputNode(n.data as Record<string, unknown> | undefined),
    [],
  );

  const nodes = useMemo(() => {
    let out = allNodes;
    if (!showAccessGrants) out = out.filter((n) => n.type !== "stateContainer");
    if (!showAnnotations)
      out = out.filter((n) => !ANNOTATION_NODE_TYPES.has(n.type ?? ""));
    return out;
  }, [allNodes, showAccessGrants, showAnnotations]);

  const edges = useMemo(() => {
    let out = allEdges;
    if (!showAccessGrants) out = out.filter((e) => e.type !== "accessGrant");
    if (!showViewers) {
      const viewerIds = new Set(allNodes.filter(isViewerNode).map((n) => n.id));
      if (viewerIds.size > 0) {
        out = out.filter(
          (e) => !viewerIds.has(e.source) && !viewerIds.has(e.target),
        );
      }
    }
    // Data wires render through the RoutedEdge (curved / orthogonal toggle);
    // access-grant lines keep their own dashed edge type.
    out = out.map((e) =>
      e.type === "accessGrant" ? e : { ...e, type: "routed" },
    );
    return out;
  }, [allEdges, allNodes, showAccessGrants, showViewers, isViewerNode]);

  // ── Handlers ──

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      if (mode === "root") {
        storeOnNodesChange(changes);
        return;
      }
      // Subgraph: local state + paired deletion
      setLocalNodes((nds) => {
        let updated = applyNodeChanges(changes, nds);
        const removedIds = changes
          .filter((c) => c.type === "remove")
          .map((c) => c.id);
        if (removedIds.length > 0) {
          const partnerIds = nds
            .filter((n) => removedIds.includes(n.id) && n.data.pairedWith)
            .map((n) => n.data.pairedWith as string);
          if (partnerIds.length > 0) {
            updated = updated.filter((n) => !partnerIds.includes(n.id));
          }
        }
        return updated;
      });
      setDirty(true);
    },
    [mode, storeOnNodesChange],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      if (mode === "root") {
        storeOnEdgesChange(changes);
        return;
      }
      setLocalEdges((eds) => applyEdgeChanges(changes, eds));
      setDirty(true);
    },
    [mode, storeOnEdgesChange],
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      if (mode === "root") {
        storeOnConnect(connection);
        return;
      }
      setLocalEdges((eds) => addEdge({ ...connection, animated: true }, eds));
      setDirty(true);
    },
    [mode, storeOnConnect],
  );

  // ── Node click ──

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  // ── Drag & drop ──

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const type = e.dataTransfer.getData("application/reactflow");
      if (!type || !rfInstance.current) return;

      const position = rfInstance.current.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });

      let initialData: Record<string, unknown> = {};
      const rawData = e.dataTransfer.getData("application/reactflow-data");
      if (rawData) {
        try {
          initialData = JSON.parse(rawData);
        } catch {
          /* ignore */
        }
      }

      if (mode === "root") {
        storeAddNode(type, position, initialData);
        return;
      }

      // Subgraph: local state + paired creation
      const id = genNodeId();
      const newNode: Node = {
        id,
        type,
        position,
        data: { label: initialData.label || type, ...initialData },
      };

      if (type === "iterIn" && initialData.paired) {
        // Two-sided loop: seeds → iterIn(left/init) → body → iterOut → iterIn.
        // iterIn's run-start inputs are declared via iterIn.data.initPorts;
        // only iterOut is created as a partner. iterIn keeps a back-pointer
        // to iterOut via pairedWith for the partner-deletion logic. No canvas
        // wires between the pivots — the iterOut→iterIn transfer is
        // executor-internal.
        const outId = genNodeId();
        const outNode: Node = {
          id: outId,
          type: "iterOut",
          position: { x: position.x + 500, y: position.y },
          data: { label: "Iter Out", pairedWith: id },
        };
        newNode.data = {
          ...newNode.data,
          label: "Iter In",
          pairedWith: outId,
          initPorts: [],
        };
        setLocalNodes((nds) => [...nds, newNode, outNode]);
      } else {
        setLocalNodes((nds) => [...nds, newNode]);
      }
      setDirty(true);
    },
    [mode, storeAddNode],
  );

  // ── Subgraph save/reset ──

  const handleSave = useCallback(() => {
    if (!onSave || !innerGraph) return;
    onSave({
      ...innerGraph,
      nodes: fromFlowNodes(localNodes),
      edges: fromFlowEdges(localEdges),
    });
  }, [onSave, innerGraph, localNodes, localEdges]);

  const handleReset = useCallback(() => {
    if (!innerGraph) return;
    setLocalNodes(toFlowNodes(innerGraph.nodes));
    setLocalEdges(toFlowEdges(innerGraph.edges));
    setDirty(false);
  }, [innerGraph]);

  // ── Render ──

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Subgraph toolbar */}
      {mode === "subgraph" && innerGraph && (
        <div className="flex items-center justify-between border-b border-gray-800 bg-gray-900 px-4 py-1.5">
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              {innerGraph.name || innerGraph.presetId || "Subgraph"} — max{" "}
              {innerGraph.step_budget ?? 500} iterations
            </span>
            <span className="text-xs text-gray-600">
              {localNodes.length} nodes, {localEdges.length} edges
            </span>
            {dirty && <span className="text-xs text-yellow-500">unsaved</span>}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              disabled={!dirty}
              className="flex items-center gap-1 rounded border border-gray-700 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-30"
            >
              <RotateCcw size={12} /> Reset
            </button>
            <button
              onClick={handleSave}
              disabled={!dirty}
              className="flex items-center gap-1 rounded bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-500 disabled:opacity-30"
            >
              <Save size={12} /> Save & Back
            </button>
          </div>
        </div>
      )}

      {/* Canvas */}
      <div style={{ flex: 1, position: "relative" }}>
        <div
          style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onDragOver={onDragOver}
            onDrop={onDrop}
            onInit={(instance) => {
              rfInstance.current = instance;
              (window as unknown as Record<string, unknown>).__rf = instance;
            }}
            nodeTypes={proxiedNodeTypes}
            edgeTypes={customEdgeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.1}
            maxZoom={5}
            defaultEdgeOptions={{ animated: true }}
            className={`bg-gray-950${showAccessGrants ? " show-state-edges" : ""}`}
            style={{ width: "100%", height: "100%" }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="#374151"
            />
            <Controls />
            <MiniMap
              style={{ background: "#111827" }}
              nodeColor={(node) => {
                const schema = (
                  node.data as
                    | {
                        _schema?: {
                          category?: string;
                          ui_config?: { color?: string };
                        };
                      }
                    | undefined
                )?._schema;
                const uiColor = schema?.ui_config?.color;
                const category = schema?.category ?? "custom";
                const normalizedCat = category.startsWith("server:")
                  ? "server"
                  : category;
                const style =
                  (uiColor && COLOR_MAP[uiColor]) ||
                  CATEGORY_STYLES[normalizedCat] ||
                  DEFAULT_STYLE;
                return style.handle;
              }}
              maskColor="rgba(0,0,0,0.6)"
            />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}
