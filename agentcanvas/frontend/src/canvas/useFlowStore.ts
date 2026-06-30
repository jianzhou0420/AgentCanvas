/** Zustand store for React Flow canvas state — tab-aware unified canvas model.
 *
 * Each tab owns its own nodes/edges/canvasStack/nodeOutputs. The top-level
 * fields (nodes, edges, etc.) are backward-compatible PROJECTIONS of the
 * active tab — existing components that read s.nodes continue to work.
 */

import { create } from "zustand";
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from "@xyflow/react";
import { defaultNodes, defaultEdges, TEMPLATES } from "./defaultGraph";
import { isCompatibleWireConnection } from "./nodes/agentloop/inner/layouts/layoutUtils";
import {
  fromFlowNodes,
  fromFlowEdges,
  toFlowNodes,
  toFlowEdges,
  toFlowContainerNodes,
  toFlowAccessGrants,
  fromFlowContainerNodes,
  fromFlowAccessGrants,
} from "./graphConversion";
import {
  FRONTEND_ONLY_TYPES,
  OUTPUT_NODE_TYPES,
  isOutputNode,
} from "./unifiedNodeTypes";
import type { GraphDefinition } from "./types";
import type { NodeInstanceData } from "../types";
import { registerFlowStoreBridge } from "./flowStoreRef";
import {
  resolveInstancePorts,
  synthesizeIterInPortsForId,
} from "./portResolution";

/** Split the well-known "graph_state" container out of a container list
 *  for frontend UI purposes.  Returns `[graphStateContainer, remaining]`. */
function splitGraphState(
  containers: import("./types").ContainerDef[] | undefined,
): [import("./types").ContainerDef | null, import("./types").ContainerDef[]] {
  if (!containers || containers.length === 0) return [null, []];
  const gs = containers.find((c) => c.id === "graph_state") ?? null;
  const rest = containers.filter((c) => c.id !== "graph_state");
  return [gs, rest];
}

/** Merge the frontend tab.graphState slice back into a container list
 *  as the well-known "graph_state" container. */
export function mergeGraphState(
  containers: import("./types").ContainerDef[],
  graphState: import("./types").ContainerDef | null,
): import("./types").ContainerDef[] {
  if (!graphState) return containers.filter((c) => c.id !== "graph_state");
  const filtered = containers.filter((c) => c.id !== "graph_state");
  return [...filtered, { ...graphState, id: "graph_state" }];
}

const DEFAULT_NODE_OUTPUT: NodeInstanceData = {
  status: "idle",
  currentStep: 0,
  currentRgb: "",
  currentDepth: "",
  steps: [],
  llmSteps: [],
  metrics: null,
  fields: {},
};

/* ── ID generation ── */

function genTabId(): string {
  return `tab_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export function genNodeId(type?: string): string {
  const hash = Math.random().toString(16).slice(2, 10);
  return type ? `${type}_${hash}` : `node_${hash}`;
}

/* ── Tab State ── */

export interface TabState {
  id: string;
  title: string;
  graphId: string | null; // null = unsaved
  description: string; // graph metadata, preserved across in-place saves
  dirty: boolean;
  nodes: Node[];
  edges: Edge[];
  nodeConfigs: Record<string, Record<string, unknown>>;
  selectedNodeId: string | null;
  nodeOutputs: Record<string, NodeInstanceData>;
  activeExecutions: Record<
    string,
    { agentNodeId: string; outputNodeIds: string[] }
  >;
  canvasStack: Array<{ nodeId: string; label: string }>;
  activeSubgraphNodeId: string | null;
  /** The well-known "graph_state" container — split out of `containers` on
   *  load so that the banner / panel UI can render it distinctly from the
   *  other state containers.  Merged back into `containers` on save. */
  graphState: import("./types").ContainerDef | null;
  /** Live preview data for the graph_state container (from WS nav_step.containers). */
  graphStatePreview: Record<string, Record<string, unknown>> | null;
  /** Per-episode iteration cap for the graph executor (editable from toolbar). */
  step_budget: number;
}

function createTabState(
  id: string,
  title: string,
  nodes: Node[] = [],
  edges: Edge[] = [],
  graphId: string | null = null,
  description: string = "",
): TabState {
  return {
    id,
    title,
    graphId,
    description,
    dirty: false,
    nodes,
    edges,
    nodeConfigs: {},
    selectedNodeId: null,
    nodeOutputs: {},
    activeExecutions: {},
    canvasStack: [],
    activeSubgraphNodeId: null,
    graphState: null,
    graphStatePreview: null,
    step_budget: 500,
  };
}

/* ── Helper: update active tab + sync projection ── */

type TabUpdater = (tab: TabState) => Partial<TabState>;

function updateActiveTab(
  s: FlowStore,
  updater: TabUpdater,
): Partial<FlowStore> {
  const tab = s.tabs[s.activeTabId];
  if (!tab) return {};
  const patch = updater(tab);
  const updated = { ...tab, ...patch };
  return {
    tabs: { ...s.tabs, [s.activeTabId]: updated },
    // Sync projections for any fields that changed
    ...(patch.nodes !== undefined ? { nodes: updated.nodes } : {}),
    ...(patch.edges !== undefined ? { edges: updated.edges } : {}),
    ...(patch.nodeConfigs !== undefined
      ? { nodeConfigs: updated.nodeConfigs }
      : {}),
    ...(patch.selectedNodeId !== undefined
      ? { selectedNodeId: updated.selectedNodeId }
      : {}),
    ...(patch.nodeOutputs !== undefined
      ? { nodeOutputs: updated.nodeOutputs }
      : {}),
    ...(patch.activeExecutions !== undefined
      ? { activeExecutions: updated.activeExecutions }
      : {}),
    ...(patch.canvasStack !== undefined
      ? { canvasStack: updated.canvasStack }
      : {}),
    ...(patch.activeSubgraphNodeId !== undefined
      ? { activeSubgraphNodeId: updated.activeSubgraphNodeId }
      : {}),
    ...(patch.graphState !== undefined
      ? { graphState: updated.graphState }
      : {}),
    ...(patch.graphStatePreview !== undefined
      ? { graphStatePreview: updated.graphStatePreview }
      : {}),
  };
}

/* ── Store Interface ── */

interface FlowStore {
  // ── Tab management ──
  tabs: Record<string, TabState>;
  activeTabId: string;
  tabOrder: string[];
  newTab: (title?: string) => string;
  closeTab: (tabId: string) => void;
  setActiveTab: (tabId: string) => void;
  openGraphInTab: (
    graphId: string,
    graph: GraphDefinition,
    nodeSchemas?: Record<string, unknown>[],
  ) => void;
  getActiveTab: () => TabState | undefined;
  updateTabMeta: (
    tabId: string,
    patch: Partial<
      Pick<TabState, "title" | "graphId" | "dirty" | "description">
    >,
  ) => void;

  // ── Backward-compatible projections of active tab ──
  nodes: Node[];
  edges: Edge[];
  nodeConfigs: Record<string, Record<string, unknown>>;
  selectedNodeId: string | null;
  nodeOutputs: Record<string, NodeInstanceData>;
  activeExecutions: Record<
    string,
    { agentNodeId: string; outputNodeIds: string[] }
  >;
  canvasStack: Array<{ nodeId: string; label: string }>;
  activeSubgraphNodeId: string | null;

  // ── Mutation methods (operate on active tab) ──
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  setNodeConfig: (nodeId: string, key: string, value: unknown) => void;
  getNodeConfig: (nodeId: string) => Record<string, unknown>;
  setSelectedNodeId: (id: string | null) => void;
  /** Update a node's data fields. Works in both root and subgraph mode. */
  updateNodeData: (nodeId: string, patch: Record<string, unknown>) => void;
  /** Remove any edges matching the predicate from the active tab. */
  removeEdgesWhere: (predicate: (edge: Edge) => boolean) => void;
  /** Recompute iterIn.data.ports + _schema.output_ports for one iterIn node,
   *  or all of them if nodeId is omitted. Call after Init/iterOut ports change
   *  or a canvas edge targeting an iterIn is added/removed. */
  resyncIterInPorts: (iterInId?: string) => void;
  /** Registered by UnifiedGraphEditor in subgraph mode for PropertiesPanel access. */
  _subgraphNodeUpdater:
    | ((nodeId: string, patch: Record<string, unknown>) => void)
    | null;
  setSubgraphNodeUpdater: (
    fn: ((nodeId: string, patch: Record<string, unknown>) => void) | null,
  ) => void;
  /** The currently visible nodes (root or subgraph). Set by UnifiedGraphEditor. */
  visibleNodes: import("@xyflow/react").Node[];
  setVisibleNodes: (nodes: import("@xyflow/react").Node[]) => void;
  addNode: (
    type: string,
    position: { x: number; y: number },
    initialData?: Record<string, unknown>,
  ) => void;
  removeNode: (nodeId: string) => void;
  resetToDefault: () => void;
  loadTemplate: (templateId: string) => void;
  loadGraph: (
    graph: GraphDefinition,
    nodeSchemas?: Record<string, unknown>[],
  ) => void;

  // ── Execution ──
  getGraphForExecution: () => GraphDefinition | null;
  getOutputNodeIds: () => string[];
  startExecution: (executionId: string) => void;
  updateNodeOutput: (nodeId: string, update: Partial<NodeInstanceData>) => void;
  /** Update node output in a specific tab (for background tab WS routing). */
  updateNodeOutputInTab: (
    tabId: string,
    nodeId: string,
    update: Partial<NodeInstanceData>,
  ) => void;

  // ── Subgraph navigation ──
  pushCanvas: (nodeId: string, label: string) => void;
  popCanvas: () => void;
  goToDepth: (depth: number) => void;
  openSubgraph: (nodeId: string) => void;
  closeSubgraph: () => void;

  // ── State edge visibility ──
  showStateEdges: boolean;
  toggleStateEdges: () => void;

  // ── Viewer visibility (render-only filter; edges stay regular data edges) ──
  showViewers: boolean;
  toggleViewers: () => void;

  // ── Annotation visibility (render-only filter for note + future sticker/link) ──
  showAnnotations: boolean;
  toggleAnnotations: () => void;

  // ── Graph state ──
  graphState: import("./types").ContainerDef | null;
  graphStatePreview: Record<string, Record<string, unknown>> | null;
  setStepBudget: (n: number) => void;
  setGraphState: (gs: import("./types").ContainerDef | null) => void;
  updateGraphStatePreview: (
    preview: Record<string, Record<string, unknown>> | null,
  ) => void;
  // Live previews of all containers during a run (home + nodeset-owned),
  // keyed by container id. Pushed from the WS ``nav_step`` event. Top-level
  // (run telemetry, not per-tab) — only one run is active at a time.
  containersLive: Record<
    string,
    {
      label: string;
      owner: string;
      states: Record<string, Record<string, unknown>>;
    }
  > | null;
  setContainersLive: (
    payload: Record<
      string,
      {
        label: string;
        owner: string;
        states: Record<string, Record<string, unknown>>;
      }
    > | null,
  ) => void;
}

/* ── Initial tab ── */

const initialTabId = genTabId();
const initialTab = createTabState(initialTabId, "Untitled");

/* ── Store ── */

export const useFlowStore = create<FlowStore>((set, get) => ({
  // ── Tab state ──
  tabs: { [initialTabId]: initialTab },
  activeTabId: initialTabId,
  tabOrder: [initialTabId],

  // ── Projections (mirroring active tab) ──
  nodes: initialTab.nodes,
  edges: initialTab.edges,
  nodeConfigs: initialTab.nodeConfigs,
  selectedNodeId: initialTab.selectedNodeId,
  nodeOutputs: initialTab.nodeOutputs,
  activeExecutions: initialTab.activeExecutions,
  canvasStack: initialTab.canvasStack,
  activeSubgraphNodeId: initialTab.activeSubgraphNodeId,
  graphState: initialTab.graphState,
  graphStatePreview: initialTab.graphStatePreview,
  containersLive: null,

  // ── Tab lifecycle ──

  newTab: (title) => {
    const id = genTabId();
    const tab = createTabState(id, title || "Untitled");
    set((s) => ({
      tabs: { ...s.tabs, [id]: tab },
      tabOrder: [...s.tabOrder, id],
      activeTabId: id,
      // Swap projections to new tab
      nodes: tab.nodes,
      edges: tab.edges,
      nodeConfigs: tab.nodeConfigs,
      selectedNodeId: tab.selectedNodeId,
      nodeOutputs: tab.nodeOutputs,
      activeExecutions: tab.activeExecutions,
      canvasStack: tab.canvasStack,
      activeSubgraphNodeId: tab.activeSubgraphNodeId,
    }));
    return id;
  },

  closeTab: (tabId) => {
    const s = get();
    const order = s.tabOrder.filter((id) => id !== tabId);
    // If closing the last tab, create a new empty one
    if (order.length === 0) {
      const newId = genTabId();
      const newTabState = createTabState(newId, "Untitled");
      set({
        tabs: { [newId]: newTabState },
        tabOrder: [newId],
        activeTabId: newId,
        nodes: newTabState.nodes,
        edges: newTabState.edges,
        nodeConfigs: newTabState.nodeConfigs,
        selectedNodeId: null,
        nodeOutputs: {},
        activeExecutions: {},
        canvasStack: [],
        activeSubgraphNodeId: null,
      });
      return;
    }
    // Pick new active tab if closing the active one
    let newActiveId = s.activeTabId;
    if (tabId === s.activeTabId) {
      const oldIdx = s.tabOrder.indexOf(tabId);
      newActiveId = order[Math.min(oldIdx, order.length - 1)];
    }
    const { [tabId]: _, ...remainingTabs } = s.tabs;
    const activeTab = remainingTabs[newActiveId];
    set({
      tabs: remainingTabs,
      tabOrder: order,
      activeTabId: newActiveId,
      // Swap projections
      nodes: activeTab?.nodes ?? [],
      edges: activeTab?.edges ?? [],
      nodeConfigs: activeTab?.nodeConfigs ?? {},
      selectedNodeId: activeTab?.selectedNodeId ?? null,
      nodeOutputs: activeTab?.nodeOutputs ?? {},
      activeExecutions: activeTab?.activeExecutions ?? {},
      canvasStack: activeTab?.canvasStack ?? [],
      activeSubgraphNodeId: activeTab?.activeSubgraphNodeId ?? null,
      graphState: activeTab?.graphState ?? null,
      graphStatePreview: activeTab?.graphStatePreview ?? null,
    });
  },

  setActiveTab: (tabId) => {
    const tab = get().tabs[tabId];
    if (!tab) return;
    set({
      activeTabId: tabId,
      nodes: tab.nodes,
      edges: tab.edges,
      nodeConfigs: tab.nodeConfigs,
      selectedNodeId: tab.selectedNodeId,
      nodeOutputs: tab.nodeOutputs,
      activeExecutions: tab.activeExecutions,
      canvasStack: tab.canvasStack,
      activeSubgraphNodeId: tab.activeSubgraphNodeId,
      graphState: tab.graphState,
      graphStatePreview: tab.graphStatePreview,
    });
  },

  openGraphInTab: (graphId, graph, nodeSchemas) => {
    const s = get();
    // Focus existing tab if already open
    for (const tab of Object.values(s.tabs)) {
      if (tab.graphId === graphId) {
        get().setActiveTab(tab.id);
        return;
      }
    }
    // Create new tab with graph data
    const id = genTabId();
    const [gsContainer, otherContainers] = splitGraphState(graph.containers);
    const nodes = [
      ...toFlowNodes(graph.nodes),
      ...toFlowContainerNodes(otherContainers),
    ];
    const edges = [
      ...toFlowEdges(graph.edges),
      ...toFlowAccessGrants(graph.access_grants || []),
    ];

    // Inject _schema for nodeset nodes
    if (nodeSchemas && nodeSchemas.length > 0) {
      const schemaMap = new Map(
        nodeSchemas.map((s2) => [s2.type as string, s2]),
      );
      for (const node of nodes) {
        if (!node.data?._schema) {
          const rawSchema = schemaMap.get(node.type || "");
          if (rawSchema) {
            const schema = resolveInstancePorts(
              rawSchema as Record<string, unknown>,
              node.data as Record<string, unknown>,
            );
            node.data = { ...node.data, _schema: schema };
          }
        }
      }
    }
    const tab = createTabState(
      id,
      graph.name || graphId,
      nodes,
      edges,
      graphId,
      graph.description || "",
    );
    tab.graphState = gsContainer;
    tab.step_budget = graph.step_budget ?? 500;
    set({
      tabs: { ...s.tabs, [id]: tab },
      tabOrder: [...s.tabOrder, id],
      activeTabId: id,
      nodes: tab.nodes,
      edges: tab.edges,
      nodeConfigs: tab.nodeConfigs,
      selectedNodeId: null,
      nodeOutputs: {},
      activeExecutions: {},
      canvasStack: [],
      activeSubgraphNodeId: null,
      graphState: tab.graphState,
      graphStatePreview: null,
    });
  },

  getActiveTab: () => get().tabs[get().activeTabId],

  updateTabMeta: (tabId, patch) =>
    set((s) => {
      const tab = s.tabs[tabId];
      if (!tab) return {};
      return { tabs: { ...s.tabs, [tabId]: { ...tab, ...patch } } };
    }),

  // ── Mutation methods (operate on active tab via projection sync) ──

  onNodesChange: (changes) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        nodes: applyNodeChanges(changes, tab.nodes),
        dirty: true,
      })),
    ),

  onEdgesChange: (changes) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        edges: applyEdgeChanges(changes, tab.edges),
        dirty: true,
      })),
    ),

  onConnect: (connection) => {
    const sourceNode = get().nodes.find((n) => n.id === connection.source);
    const targetNode = get().nodes.find((n) => n.id === connection.target);
    if (!sourceNode || !targetNode) return;

    // Wire-type compatibility check (ADR-027).  Look up the declared
    // wire_type on each end from the node's _schema and reject lossy or
    // mismatched combinations.  Equal / ANY / T→LIST[T] are allowed.
    // Skipped for access grants below.
    const pickPortType = (
      node: typeof sourceNode,
      handle: string | null | undefined,
      side: "input_ports" | "output_ports",
    ): string | undefined => {
      if (!handle || handle === "__state__") return undefined;
      const schema = (node.data as Record<string, unknown> | undefined)
        ?._schema as
        | { input_ports?: unknown; output_ports?: unknown }
        | undefined;
      const ports =
        (schema?.[side] as
          | Array<{ name: string; wire_type: string }>
          | undefined) ?? [];
      return ports.find((p) => p.name === handle)?.wire_type;
    };
    const sourceIsContainerLike =
      sourceNode.type === "stateContainer" ||
      targetNode.type === "stateContainer" ||
      connection.sourceHandle === "__state__" ||
      connection.targetHandle === "__state__";
    if (!sourceIsContainerLike) {
      const srcType = pickPortType(
        sourceNode,
        connection.sourceHandle,
        "output_ports",
      );
      const tgtType = pickPortType(
        targetNode,
        connection.targetHandle,
        "input_ports",
      );
      if (!isCompatibleWireConnection(srcType, tgtType)) {
        // Rejected — drop the drag silently (red snap is handled by React Flow).
        return;
      }
    }

    // Access grant: when either end is a stateContainer, or connection uses __state__ handle
    const isAccessGrant =
      sourceNode.type === "stateContainer" ||
      targetNode.type === "stateContainer" ||
      connection.sourceHandle === "__state__" ||
      connection.targetHandle === "__state__";
    if (isAccessGrant) {
      // Guard: an access grant requires exactly one container and one regular node
      const sourceIsContainer = sourceNode.type === "stateContainer";
      const targetIsContainer = targetNode.type === "stateContainer";
      if (sourceIsContainer === targetIsContainer) return;

      // Normalize direction: source = regular node, target = stateContainer
      const nodeId = sourceIsContainer
        ? connection.target!
        : connection.source!;
      const containerId = sourceIsContainer
        ? connection.source!
        : connection.target!;

      const grantEdge: Edge = {
        id: `ag_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        source: nodeId,
        target: containerId,
        sourceHandle: "__state__",
        targetHandle: "state",
        type: "accessGrant",
      };
      set((s) =>
        updateActiveTab(s, (tab) => ({
          edges: [...tab.edges, grantEdge],
          dirty: true,
        })),
      );
      return;
    }

    set((s) =>
      updateActiveTab(s, (tab) => ({
        edges: addEdge(connection, tab.edges),
        dirty: true,
      })),
    );
  },

  setNodeConfig: (nodeId, key, value) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        nodeConfigs: {
          ...tab.nodeConfigs,
          [nodeId]: { ...(tab.nodeConfigs[nodeId] || {}), [key]: value },
        },
        dirty: true,
      })),
    ),

  getNodeConfig: (nodeId) =>
    get().tabs[get().activeTabId]?.nodeConfigs[nodeId] || {},

  setSelectedNodeId: (id) =>
    set((s) => updateActiveTab(s, () => ({ selectedNodeId: id }))),

  _subgraphNodeUpdater: null,
  setSubgraphNodeUpdater: (fn) => set({ _subgraphNodeUpdater: fn }),
  visibleNodes: initialTab.nodes,
  setVisibleNodes: (nodes) => set({ visibleNodes: nodes }),

  updateNodeData: (nodeId, patch) => {
    const s = get();
    // If in subgraph mode, delegate to the registered updater
    if (s._subgraphNodeUpdater) {
      s._subgraphNodeUpdater(nodeId, patch);
      return;
    }
    // Root mode: update tab nodes directly
    set((prev) =>
      updateActiveTab(prev, (tab) => ({
        nodes: tab.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n,
        ),
        dirty: true,
      })),
    );
  },

  removeEdgesWhere: (predicate) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        edges: tab.edges.filter((e) => !predicate(e)),
        dirty: true,
      })),
    ),

  resyncIterInPorts: (iterInId) =>
    set((s) =>
      updateActiveTab(s, (tab) => {
        const targets = iterInId
          ? [iterInId]
          : tab.nodes.filter((n) => n.type === "iterIn").map((n) => n.id);
        if (targets.length === 0) return {};
        const inputNodes = tab.nodes.map((n) => ({
          id: n.id,
          type: n.type,
          data: n.data as Record<string, unknown>,
        }));
        const newNodes = tab.nodes.map((n) => {
          if (!targets.includes(n.id) || n.type !== "iterIn") return n;
          const ports = synthesizeIterInPortsForId(n.id, inputNodes, tab.edges);
          const prevSchema = (n.data as Record<string, unknown>)._schema as
            | Record<string, unknown>
            | undefined;
          const nextSchema = prevSchema
            ? {
                ...prevSchema,
                output_ports: ports.map((p) => {
                  const entry: Record<string, unknown> = {
                    name: p.name,
                    wire_type: p.wire_type,
                    description: "",
                    optional: true,
                  };
                  if (p.persist !== undefined) entry.persist = p.persist;
                  const pp = p as {
                    origin?: string;
                    writer_name?: string;
                  };
                  if (pp.origin !== undefined) entry.origin = pp.origin;
                  if (pp.writer_name !== undefined)
                    entry.writer_name = pp.writer_name;
                  return entry;
                }),
              }
            : prevSchema;
          return {
            ...n,
            data: {
              ...n.data,
              ports,
              ...(nextSchema ? { _schema: nextSchema } : {}),
            },
          };
        });
        return { nodes: newNodes, dirty: true };
      }),
    ),

  addNode: (type, position, initialData) => {
    const id = genNodeId();
    const data = initialData || {};
    // Auto-populate label from subgraph name for composite nodes
    if (
      !data.label &&
      data.subgraph &&
      (data.subgraph as Record<string, unknown>).name
    ) {
      data.label = (data.subgraph as Record<string, unknown>).name;
    }
    const node: Node = { id, type, position, data };
    set((s) =>
      updateActiveTab(s, (tab) => ({
        nodes: [...tab.nodes, node],
        dirty: true,
      })),
    );
  },

  removeNode: (nodeId) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        nodes: tab.nodes.filter((n) => n.id !== nodeId),
        edges: tab.edges.filter(
          (e) => e.source !== nodeId && e.target !== nodeId,
        ),
        dirty: true,
      })),
    ),

  resetToDefault: () =>
    set((s) =>
      updateActiveTab(s, () => ({
        nodes: [...defaultNodes],
        edges: [...defaultEdges],
        nodeConfigs: {},
        dirty: false,
      })),
    ),

  loadTemplate: (templateId) => {
    const tmpl = TEMPLATES.find((t) => t.id === templateId);
    if (!tmpl) return;
    set((s) =>
      updateActiveTab(s, () => ({
        nodes: [...tmpl.nodes],
        edges: [...tmpl.edges],
        nodeConfigs: {},
        canvasStack: [],
        activeSubgraphNodeId: null,
        dirty: false,
        title: tmpl.name,
      })),
    );
  },

  loadGraph: (graph, nodeSchemas) => {
    const nodes = toFlowNodes(graph.nodes);
    const edges = toFlowEdges(graph.edges);

    // Split graph_state out of containers; the rest become canvas nodes.
    const [gsContainer, otherContainers] = splitGraphState(graph.containers);
    const containerNodes = toFlowContainerNodes(otherContainers);
    const grantEdges = toFlowAccessGrants(graph.access_grants || []);

    // Inject _schema from backend node schemas so GenericBlockRenderer
    // can render ports (Handles) for nodeset nodes.
    if (nodeSchemas && nodeSchemas.length > 0) {
      const schemaMap = new Map(nodeSchemas.map((s) => [s.type as string, s]));
      for (const node of nodes) {
        if (!node.data?._schema) {
          const rawSchema = schemaMap.get(node.type || "");
          if (rawSchema) {
            const schema = resolveInstancePorts(
              rawSchema as Record<string, unknown>,
              node.data as Record<string, unknown>,
            );
            node.data = { ...node.data, _schema: schema };
          }
        }
      }
    }

    set((s) =>
      updateActiveTab(s, () => ({
        nodes: [...nodes, ...containerNodes],
        edges: [...edges, ...grantEdges],
        nodeConfigs: {},
        canvasStack: [],
        activeSubgraphNodeId: null,
        graphState: gsContainer,
        graphStatePreview: null,
        step_budget: graph.step_budget ?? 500,
        dirty: false,
      })),
    );
  },

  // ── Execution ──

  getGraphForExecution: () => {
    const { nodes, edges } = get();
    // Separate container nodes and access-grant edges from execution nodes
    const containerNodes = nodes.filter((n) => n.type === "stateContainer");
    const execNodes = nodes.filter(
      (n) => !FRONTEND_ONLY_TYPES.has(n.type || ""),
    );
    if (execNodes.length === 0) return null;
    const execNodeIds = new Set(execNodes.map((n) => n.id));
    const dataEdges = edges.filter(
      (e) =>
        e.type !== "accessGrant" &&
        execNodeIds.has(e.source) &&
        execNodeIds.has(e.target),
    );
    const grantEdges = edges.filter((e) => e.type === "accessGrant");
    const tab = get().tabs[get().activeTabId];
    // Merge the frontend-only tab.graphState slice back into `containers`
    // as the well-known "graph_state" container (ADR-026).
    const containers = mergeGraphState(
      fromFlowContainerNodes(containerNodes),
      tab?.graphState || null,
    );
    return {
      name: tab?.title || "Root Graph",
      description: "",
      nodes: fromFlowNodes(execNodes),
      edges: fromFlowEdges(dataEdges),
      containers,
      access_grants: fromFlowAccessGrants(grantEdges),
      step_budget: tab?.step_budget ?? 500,
    };
  },

  getOutputNodeIds: () => {
    const { nodes } = get();
    return nodes
      .filter(
        (n) =>
          isOutputNode(n.data as Record<string, unknown>) ||
          OUTPUT_NODE_TYPES.has(n.type || ""),
      )
      .map((n) => n.id);
  },

  startExecution: (executionId) => {
    const outputNodeIds = get().getOutputNodeIds();
    set((s) => {
      const tab = s.tabs[s.activeTabId];
      if (!tab) return {};
      const nodeOutputs = { ...tab.nodeOutputs };
      for (const nid of outputNodeIds) {
        nodeOutputs[nid] = { ...DEFAULT_NODE_OUTPUT };
      }
      const activeExecutions = {
        ...tab.activeExecutions,
        [executionId]: { agentNodeId: "__root__", outputNodeIds },
      };
      const updated = { ...tab, nodeOutputs, activeExecutions };
      return {
        tabs: { ...s.tabs, [s.activeTabId]: updated },
        nodeOutputs,
        activeExecutions,
      };
    });
  },

  updateNodeOutput: (nodeId, update) =>
    set((s) =>
      updateActiveTab(s, (tab) => ({
        nodeOutputs: {
          ...tab.nodeOutputs,
          [nodeId]: {
            ...(tab.nodeOutputs[nodeId] || DEFAULT_NODE_OUTPUT),
            ...update,
          },
        },
      })),
    ),

  updateNodeOutputInTab: (tabId, nodeId, update) =>
    set((s) => {
      const tab = s.tabs[tabId];
      if (!tab) return {};
      const newOutputs = {
        ...tab.nodeOutputs,
        [nodeId]: {
          ...(tab.nodeOutputs[nodeId] || DEFAULT_NODE_OUTPUT),
          ...update,
        },
      };
      const updated = { ...tab, nodeOutputs: newOutputs };
      const result: Partial<FlowStore> = {
        tabs: { ...s.tabs, [tabId]: updated },
      };
      // Sync projection if this is the active tab
      if (tabId === s.activeTabId) {
        result.nodeOutputs = newOutputs;
      }
      return result;
    }),

  // ── Subgraph navigation ──

  pushCanvas: (nodeId, label) =>
    set((s) => {
      const tab = s.tabs[s.activeTabId];
      if (!tab) return {};
      const stack = [...tab.canvasStack, { nodeId, label }];
      const updated = {
        ...tab,
        canvasStack: stack,
        activeSubgraphNodeId: stack[stack.length - 1]?.nodeId ?? null,
      };
      return {
        tabs: { ...s.tabs, [s.activeTabId]: updated },
        canvasStack: updated.canvasStack,
        activeSubgraphNodeId: updated.activeSubgraphNodeId,
      };
    }),

  popCanvas: () =>
    set((s) => {
      const tab = s.tabs[s.activeTabId];
      if (!tab) return {};
      const stack = tab.canvasStack.slice(0, -1);
      const updated = {
        ...tab,
        canvasStack: stack,
        activeSubgraphNodeId: stack[stack.length - 1]?.nodeId ?? null,
      };
      return {
        tabs: { ...s.tabs, [s.activeTabId]: updated },
        canvasStack: updated.canvasStack,
        activeSubgraphNodeId: updated.activeSubgraphNodeId,
      };
    }),

  goToDepth: (depth) =>
    set((s) => {
      const tab = s.tabs[s.activeTabId];
      if (!tab) return {};
      const stack = tab.canvasStack.slice(0, depth);
      const updated = {
        ...tab,
        canvasStack: stack,
        activeSubgraphNodeId: stack[stack.length - 1]?.nodeId ?? null,
      };
      return {
        tabs: { ...s.tabs, [s.activeTabId]: updated },
        canvasStack: updated.canvasStack,
        activeSubgraphNodeId: updated.activeSubgraphNodeId,
      };
    }),

  openSubgraph: (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    const sub = (node?.data?.innerGraph || node?.data?.subgraph) as
      | Record<string, unknown>
      | undefined;
    const label = (sub?.name as string) || "Subgraph";
    get().pushCanvas(nodeId, label);
  },

  closeSubgraph: () => get().goToDepth(0),

  // ── State edge visibility ──
  showStateEdges: false,
  toggleStateEdges: () => set((s) => ({ showStateEdges: !s.showStateEdges })),

  // ── Viewer visibility ──
  showViewers: false,
  toggleViewers: () => set((s) => ({ showViewers: !s.showViewers })),

  showAnnotations: true,
  toggleAnnotations: () =>
    set((s) => ({ showAnnotations: !s.showAnnotations })),

  setStepBudget: (n) =>
    set((s) => updateActiveTab(s, () => ({ step_budget: n }))),

  // ── Graph state (projections already set from initialTab above) ──
  setGraphState: (gs) =>
    set((s) =>
      updateActiveTab(s, () => ({
        graphState: gs,
        graphStatePreview: null,
        dirty: true,
      })),
    ),
  updateGraphStatePreview: (preview) =>
    set((s) => updateActiveTab(s, () => ({ graphStatePreview: preview }))),
  setContainersLive: (payload) => set({ containersLive: payload }),
}));

// Register bridge — scans ALL tabs for execution routing (background tabs too)
registerFlowStoreBridge({
  getActiveExecution: (executionId) => {
    const state = useFlowStore.getState();
    // Check active tab first (fast path)
    const activeTab = state.tabs[state.activeTabId];
    if (activeTab?.activeExecutions[executionId]) {
      return activeTab.activeExecutions[executionId];
    }
    // Scan all tabs for background executions
    for (const tab of Object.values(state.tabs)) {
      const exec = tab.activeExecutions[executionId];
      if (exec) return exec;
    }
    return undefined;
  },
  updateNodeOutput: (nodeId, update) => {
    // Find which tab owns this execution and update there
    const state = useFlowStore.getState();
    // Fast path: try active tab
    state.updateNodeOutput(nodeId, update);
  },
  getNodeOutput: (nodeId) => useFlowStore.getState().nodeOutputs[nodeId],
  setContainersLive: (payload) =>
    useFlowStore.getState().setContainersLive(payload),
});
