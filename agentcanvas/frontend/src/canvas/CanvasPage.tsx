/** Canvas page — tabbed workspace with explorer sidebar and graph editor. */

import {
  Component,
  useEffect,
  useState,
  useMemo,
  useCallback,
  type ReactNode,
} from "react";
import { ReactFlowProvider } from "@xyflow/react";
import {
  LayoutTemplate,
  ArrowLeft,
  ChevronRight,
  Save,
  Package,
  Wand2,
  Spline,
  Waypoints,
} from "lucide-react";

// Error boundary
class CanvasErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: { componentStack?: string }) {
    console.error("Canvas error:", error);
    // Surface to Report tab.
    void import("../errorStore").then(({ useErrorStore }) => {
      useErrorStore.getState().reportLocal({
        source: "frontend",
        severity: "error",
        code: "REACT_CRASH",
        title: `React crash: ${error.message}`,
        message: error.message,
        details: {
          stack: error.stack || "",
          componentStack: info?.componentStack || "",
        },
      });
    });
  }
  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-1 items-center justify-center bg-gray-950 p-8 text-center">
          <div>
            <div className="mb-2 text-lg font-semibold text-red-400">
              Canvas Error
            </div>
            <div className="mb-4 text-sm text-gray-400">
              {this.state.error.message}
            </div>
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

import UnifiedGraphEditor from "./UnifiedGraphEditor";
import ExplorerPanel from "./panels/ExplorerPanel";
import TabBar from "./panels/TabBar";
import ShortcutCheatsheet from "./ShortcutCheatsheet";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import ResizableBottomPanel from "./panels/ResizableBottomPanel";
import ResizableRightPanel from "./panels/ResizableRightPanel";
import TemplatePicker from "./panels/TemplatePicker";
import ExecutionToolbar from "./panels/ExecutionToolbar";
import EnvPanel from "./panels/EnvPanel";
import GraphStateBanner from "./panels/GraphStateBanner";
import SaveGraphDialog from "./panels/SaveGraphDialog";
import { buildUnifiedCatalog, type NodeSchema } from "./unifiedCatalog";
import { useFlowStore, mergeGraphState } from "./useFlowStore";
import { resolveInstancePorts } from "./portResolution";
import {
  fromFlowNodes,
  fromFlowEdges,
  fromFlowContainerNodes,
  fromFlowAccessGrants,
  flowToGraph,
} from "./graphConversion";
import { api } from "../api";
import type { GraphDefinition, NodeDef, ContainerDef } from "./types";

export default function CanvasPage() {
  const [showTemplatePicker, setShowTemplatePicker] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showSaveAsNodeDialog, setShowSaveAsNodeDialog] = useState(false);
  const [nodeSchemas, setNodeSchemas] = useState<NodeSchema[] | null>(null);
  const [showCheatsheet, setShowCheatsheet] = useState(false);

  useKeyboardShortcuts({
    onToggleCheatsheet: useCallback(() => setShowCheatsheet((v) => !v), []),
  });

  // Tab state
  const activeTabId = useFlowStore((s) => s.activeTabId);
  const routingMode = useFlowStore((s) => s.routingMode);
  const toggleRoutingMode = useFlowStore((s) => s.toggleRoutingMode);

  // Canvas stack navigation (ADR-006) — per-tab via projection
  const canvasStack = useFlowStore((s) => s.canvasStack);
  const popCanvas = useFlowStore((s) => s.popCanvas);
  const goToDepth = useFlowStore((s) => s.goToDepth);

  // Resolve the active node at the top of the stack
  const activeSubgraphNodeId = useFlowStore((s) => s.activeSubgraphNodeId);
  const activeNode = useFlowStore((s) =>
    s.activeSubgraphNodeId
      ? s.nodes.find((n) => n.id === s.activeSubgraphNodeId)
      : null,
  );
  const innerGraph = (activeNode?.data?.innerGraph ||
    activeNode?.data?.subgraph) as GraphDefinition | undefined;

  // EnvPanel manages its own state — queries /api/env-panels on
  // mount and listens for nodesets-changed itself. No env-detection needed.

  // Discover tools and node schemas from backend
  const refreshSchemas = useCallback(() => {
    api
      .getNodeSchemas()
      .then((schemas) => {
        setNodeSchemas(schemas as unknown as NodeSchema[]);
        // Patch _schema on existing canvas nodes so dropdowns refresh
        const schemaMap = new Map(
          (schemas as Array<{ type: string }>).map((s) => [s.type, s]),
        );
        const { nodes } = useFlowStore.getState();
        const patched = nodes.map((n) => {
          const rawSchema = schemaMap.get(n.type || "");
          if (rawSchema && n.data?._schema) {
            const schema = resolveInstancePorts(
              rawSchema as Record<string, unknown>,
              n.data as Record<string, unknown>,
            );
            return { ...n, data: { ...n.data, _schema: schema } };
          }
          return n;
        });
        useFlowStore.setState({ nodes: patched });
      })
      .catch(() => setNodeSchemas(null));
  }, []);

  useEffect(() => {
    refreshSchemas();
  }, [refreshSchemas]);

  // Re-fetch schemas when profiles or nodesets change
  useEffect(() => {
    const handler = () => refreshSchemas();
    window.addEventListener("profiles-changed", handler);
    window.addEventListener("nodesets-changed", handler);
    return () => {
      window.removeEventListener("profiles-changed", handler);
      window.removeEventListener("nodesets-changed", handler);
    };
  }, [refreshSchemas]);

  // Unified catalog
  const catalog = useMemo(
    () => buildUnifiedCatalog(nodeSchemas),
    [nodeSchemas],
  );

  // Save inner graph back to node data
  const handleSaveInnerGraph = useCallback(
    (updated: GraphDefinition) => {
      if (!activeSubgraphNodeId) return;
      const { nodes } = useFlowStore.getState();
      const updatedNodes = nodes.map((n) =>
        n.id === activeSubgraphNodeId
          ? {
              ...n,
              data: { ...n.data, innerGraph: updated, subgraph: updated },
            }
          : n,
      );
      useFlowStore.setState({ nodes: updatedNodes });
      popCanvas();
    },
    [activeSubgraphNodeId, popCanvas],
  );

  // ── Save (in-place if possible, else fall back to Save-As dialog) ──
  //
  // Standard IDE Ctrl+S semantics: if the active tab corresponds to an
  // already-saved graph, PUT the current canvas to its existing path with
  // no prompts. Only open the dialog when there's no on-disk artifact yet
  // (new tab from a template) or we're inside a subgraph (saving a
  // composite is a different flow handled by the dialog).
  const handleSaveCurrent = useCallback(async () => {
    const state = useFlowStore.getState();
    const tab = state.tabs[state.activeTabId];
    if (!tab) return;

    if (state.canvasStack.length > 0 || !tab.graphId) {
      setShowSaveDialog(true);
      return;
    }

    try {
      // Use flowToGraph to properly split state-container nodes out of the
      // node list and access-grant edges out of the edge list, then merge
      // the tab's graph_state slice back in (it's held outside `state.nodes`
      // because the canvas renders it as a banner, not a stateContainer node).
      const flat = flowToGraph(state.nodes, state.edges, {
        name: tab.title,
        description: tab.description,
        step_budget: tab.step_budget,
      });
      const containers = mergeGraphState(flat.containers || [], tab.graphState);
      await api.updateGraph(tab.graphId, {
        name: flat.name,
        description: flat.description,
        nodes: flat.nodes as unknown[],
        edges: flat.edges as unknown[],
        containers: containers as unknown[],
        access_grants: (flat.access_grants || []) as unknown[],
        step_budget: flat.step_budget,
      });
      state.updateTabMeta(tab.id, { dirty: false });
      // The Explorer sidebar caches `allGraphs` from a single mount-time
      // `listGraphs()` and only re-fetches via the global hook below.
      // Without this, double-clicking the entry after an in-place save
      // reopens the pre-save snapshot from the stale cache.
      const refresh = (window as unknown as Record<string, unknown>)
        .__refreshSavedGraphs as (() => void) | undefined;
      if (refresh) refresh();
    } catch (err) {
      console.error("In-place save failed, falling back to dialog:", err);
      setShowSaveDialog(true);
    }
  }, []);

  // ── Auto Layout ──
  const handleAutoLayout = useCallback(async () => {
    const state = useFlowStore.getState();
    const { nodes, edges } = state;

    // Split containers vs regular nodes, data vs access-grant edges
    const containerFlowNodes = nodes.filter((n) => n.type === "stateContainer");
    const regularNodes = nodes.filter((n) => n.type !== "stateContainer");
    const dataEdges = edges.filter((e) => e.type !== "accessGrant");
    const grantEdges = edges.filter((e) => e.type === "accessGrant");

    const tab = state.tabs[state.activeTabId];
    const graph = {
      name: tab?.title || "Untitled",
      nodes: fromFlowNodes(regularNodes),
      edges: fromFlowEdges(dataEdges),
      containers: fromFlowContainerNodes(containerFlowNodes),
      access_grants: fromFlowAccessGrants(grantEdges),
    };

    // Real rendered sizes so the backend spaces columns by width and rows by
    // height — otherwise wide nodes (long type names, many ports) overlap the
    // next fixed-pitch column.
    const dimensions: Record<string, { width: number; height: number }> = {};
    for (const n of nodes) {
      const w = n.measured?.width ?? n.width;
      const h = n.measured?.height ?? n.height;
      if (w && h) dimensions[n.id] = { width: w, height: h };
    }

    try {
      const result = await api.layoutGraph(
        graph as Record<string, unknown>,
        dimensions,
      );
      const posMap = new Map<string, { x: number; y: number }>();
      for (const n of (result as unknown as { nodes: NodeDef[] }).nodes || []) {
        posMap.set(n.id, n.position);
      }
      for (const c of (result as unknown as { containers: ContainerDef[] })
        .containers || []) {
        posMap.set(c.id, c.position);
      }
      const updatedNodes = nodes.map((n) => {
        const newPos = posMap.get(n.id);
        return newPos ? { ...n, position: newPos } : n;
      });
      // Routing waypoints per edge id → merged into edge.data so the
      // RoutedEdge can draw the orthogonal path through the reserved channels.
      const wpMap = new Map<string, { x: number; y: number }[]>();
      for (const e of (result as unknown as {
        edges: { id?: string; waypoints?: { x: number; y: number }[] }[];
      }).edges || []) {
        if (e.id && e.waypoints) wpMap.set(e.id, e.waypoints);
      }
      const updatedEdges = edges.map((e) => ({
        ...e,
        data: { ...(e.data ?? {}), waypoints: wpMap.get(e.id) },
      }));
      // Update both tab state AND projection — otherwise onNodesChange
      // (drag) reads stale positions from the tab and snaps back.
      useFlowStore.setState((s) => {
        const tab = s.tabs[s.activeTabId];
        if (!tab) return {};
        return {
          tabs: {
            ...s.tabs,
            [s.activeTabId]: {
              ...tab,
              nodes: updatedNodes,
              edges: updatedEdges,
              dirty: true,
            },
          },
          nodes: updatedNodes,
          edges: updatedEdges,
        };
      });
    } catch (err) {
      console.error("Auto layout failed:", err);
    }
  }, []);

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === "s") {
        e.preventDefault();
        handleSaveCurrent();
      }
      if (mod && e.key === "n") {
        e.preventDefault();
        useFlowStore.getState().newTab();
      }
      if (e.ctrlKey && e.key === "Tab") {
        e.preventDefault();
        const state = useFlowStore.getState();
        const idx = state.tabOrder.indexOf(state.activeTabId);
        const next = e.shiftKey
          ? state.tabOrder[
              (idx - 1 + state.tabOrder.length) % state.tabOrder.length
            ]
          : state.tabOrder[(idx + 1) % state.tabOrder.length];
        state.setActiveTab(next);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleSaveCurrent]);

  // ── Subgraph editor view (breadcrumbs + editor inside current tab) ──
  if (canvasStack.length > 0 && innerGraph) {
    return (
      <CanvasErrorBoundary>
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <ExecutionToolbar />
          <EnvPanel />

          <TabBar />
          {/* Breadcrumb bar */}
          <div className="flex items-center gap-1 border-b border-gray-800 bg-gray-900 px-3 py-1.5">
            <button
              onClick={() => goToDepth(0)}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-blue-400 hover:bg-gray-800 hover:text-blue-300"
            >
              <ArrowLeft size={14} />
              Root
            </button>
            {canvasStack.map((frame, i) => (
              <span key={i} className="flex items-center gap-1">
                <ChevronRight size={12} className="text-gray-600" />
                {i < canvasStack.length - 1 ? (
                  <button
                    onClick={() => goToDepth(i + 1)}
                    className="rounded px-1.5 py-0.5 text-xs text-blue-400 hover:bg-gray-800 hover:text-blue-300"
                  >
                    {frame.label}
                  </button>
                ) : (
                  <span className="text-xs font-medium text-purple-400">
                    {frame.label}
                  </span>
                )}
              </span>
            ))}
            <span className="ml-2 text-[10px] text-gray-700">
              {innerGraph.nodes.length} nodes, {innerGraph.edges.length} edges
            </span>
          </div>

          <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
            <ExplorerPanel catalog={catalog} />
            <ReactFlowProvider key={`${activeTabId}-subgraph`}>
              <UnifiedGraphEditor
                mode="subgraph"
                innerGraph={innerGraph}
                nodeSchemas={
                  nodeSchemas as Array<{
                    type: string;
                    [k: string]: unknown;
                  }> | null
                }
                onSave={handleSaveInnerGraph}
                onClose={popCanvas}
              />
            </ReactFlowProvider>
            <ResizableRightPanel />
          </div>
          <ResizableBottomPanel />
        </div>
      </CanvasErrorBoundary>
    );
  }

  // ── Root canvas view (tabbed) ──
  return (
    <CanvasErrorBoundary>
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <ExecutionToolbar />
        <EnvPanel />

        <TabBar />
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <ExplorerPanel catalog={catalog} />
          <div style={{ flex: 1, position: "relative" }}>
            <div className="absolute left-2 top-2 z-10 flex gap-1">
              <button
                onClick={() => setShowTemplatePicker(true)}
                className="flex items-center gap-1 rounded border border-gray-700 bg-gray-900/90 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
              >
                <LayoutTemplate size={12} />
                Templates
              </button>
              <button
                onClick={handleSaveCurrent}
                className="flex items-center gap-1 rounded border border-gray-700 bg-gray-900/90 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
                title="Save current graph (in-place if already saved)"
              >
                <Save size={12} />
                Save
              </button>
              <button
                onClick={() => setShowSaveAsNodeDialog(true)}
                className="flex items-center gap-1 rounded border border-gray-700 bg-gray-900/90 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
                title="Archive current graph as a reusable composite node"
              >
                <Package size={12} />
                Save as Node
              </button>
            </div>
            <div className="absolute right-2 top-2 z-10 flex gap-1">
              <button
                onClick={handleAutoLayout}
                className="flex items-center gap-1 rounded border border-gray-700 bg-gray-900/90 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
                title="Auto-layout graph nodes"
              >
                <Wand2 size={12} />
                Auto Layout
              </button>
              <button
                onClick={toggleRoutingMode}
                className="flex items-center gap-1 rounded border border-gray-700 bg-gray-900/90 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
                title="Toggle wire routing: curved vs orthogonal (routes long wires around nodes; run Auto Layout first)"
              >
                {routingMode === "orthogonal" ? (
                  <Waypoints size={12} />
                ) : (
                  <Spline size={12} />
                )}
                {routingMode === "orthogonal" ? "Orthogonal" : "Curved"}
              </button>
            </div>
            {/* key={activeTabId} forces React Flow remount on tab switch */}
            <ReactFlowProvider key={activeTabId}>
              <UnifiedGraphEditor mode="root" />
            </ReactFlowProvider>
          </div>
          <ResizableRightPanel />
        </div>
        <ResizableBottomPanel />
      </div>
      {showTemplatePicker && (
        <TemplatePicker onClose={() => setShowTemplatePicker(false)} />
      )}
      {showSaveDialog && (
        <SaveGraphDialog
          onClose={() => setShowSaveDialog(false)}
          onSaved={() => {
            const refresh = (window as unknown as Record<string, unknown>)
              .__refreshSavedGraphs as (() => void) | undefined;
            if (refresh) refresh();
          }}
        />
      )}
      {showSaveAsNodeDialog && (
        <SaveGraphDialog
          kind="node"
          onClose={() => setShowSaveAsNodeDialog(false)}
          onSaved={() => {
            const refresh = (window as unknown as Record<string, unknown>)
              .__refreshSavedGraphs as (() => void) | undefined;
            if (refresh) refresh();
          }}
        />
      )}
      <ShortcutCheatsheet
        open={showCheatsheet}
        onClose={() => setShowCheatsheet(false)}
      />
    </CanvasErrorBoundary>
  );
}
