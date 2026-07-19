/** Explorer panel — VS Code-style sidebar with Graphs + Nodes sections.
 *
 * Graphs section: saved graphs from /api/graphs, double-click to open in tab.
 * Nodes section: unified catalog, drag-drop onto canvas.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Bot,
  Wrench,
  Puzzle,
  Globe,
  Image,
  Compass,
  MessageSquare,
  ScanEye,
  Brain,
  Server,
  TestTubeDiagonal,
  FileText,
  PenLine,
  Files,
  Eye,
  BrainCircuit,
  ListOrdered,
  Route,
  BarChart3,
  ChevronRight,
  ChevronDown,
  Info,
  ClipboardList,
  ArrowDownToLine,
  ArrowUpFromLine,
  FileEdit,
  GitBranch,
  MapPin,
  Radio,
  CircleStop,
  RefreshCw,
  FolderOpen,
  Folder,
  FolderPlus,
  Workflow,
  Component,
  Trash2,
  Upload,
  RotateCw,
  Loader2,
} from "lucide-react";
import clsx from "clsx";
import type { ReactNode } from "react";
import type { CatalogEntry, NodeDef } from "../types";
import type { SavedGraph } from "../../types";
import type { GraphDefinition } from "../types";
import { api } from "../../api";
import { useFlowStore } from "../useFlowStore";
import { ensureNodesetsLoaded } from "../nodesetLoader";

/* ── Shared icon maps ── */

const ICONS: Record<string, ReactNode> = {
  Bot: <Bot size={14} />,
  Wrench: <Wrench size={14} />,
  Puzzle: <Puzzle size={14} />,
  MessageSquare: <MessageSquare size={14} />,
  ScanEye: <ScanEye size={14} />,
  Brain: <Brain size={14} />,
  Server: <Server size={14} />,
  TestTubeDiagonal: <TestTubeDiagonal size={14} />,
  FileText: <FileText size={14} />,
  PenLine: <PenLine size={14} />,
  Files: <Files size={14} />,
  Eye: <Eye size={14} />,
  BrainCircuit: <BrainCircuit size={14} />,
  ListOrdered: <ListOrdered size={14} />,
  Route: <Route size={14} />,
  BarChart3: <BarChart3 size={14} />,
  ArrowDownToLine: <ArrowDownToLine size={14} />,
  ArrowUpFromLine: <ArrowUpFromLine size={14} />,
  FileEdit: <FileEdit size={14} />,
  GitBranch: <GitBranch size={14} />,
  MapPin: <MapPin size={14} />,
  Info: <Info size={14} />,
  Radio: <Radio size={14} />,
  CircleStop: <CircleStop size={14} />,
  RefreshCw: <RefreshCw size={14} />,
  Globe: <Globe size={14} />,
  FolderOpen: <FolderOpen size={14} />,
  Image: <Image size={14} />,
  Compass: <Compass size={14} />,
  ClipboardList: <ClipboardList size={14} />,
};

const CATEGORY_ICONS: Record<string, ReactNode> = {
  Graphs: <Workflow size={12} />,
  "Graph Nodes": <Component size={12} />,
  Nodes: <Wrench size={12} />,
  Environment: <Server size={12} />,
  Policy: <Brain size={12} />,
  LLM: <MessageSquare size={12} />,
  Prompt: <FileEdit size={12} />,
  History: <RefreshCw size={12} />,
  Tool: <Wrench size={12} />,
  Decision: <GitBranch size={12} />,
  Control: <CircleStop size={12} />,
  Output: <Eye size={12} />,
  Composite: <Puzzle size={12} />,
  "Agent Presets": <Bot size={12} />,
  Skill: <Puzzle size={12} />,
  Agent: <Bot size={12} />,
  Server: <Globe size={12} />,
  Custom: <Wrench size={12} />,
};

const CATEGORY_COLORS: Record<string, string> = {
  Graphs: "text-amber-400",
  "Graph Nodes": "text-indigo-400",
  Nodes: "text-cyan-400",
  Environment: "text-green-400",
  Policy: "text-blue-400",
  LLM: "text-yellow-400",
  Prompt: "text-purple-400",
  History: "text-emerald-400",
  Tool: "text-cyan-400",
  Decision: "text-pink-400",
  Control: "text-red-400",
  Output: "text-orange-400",
  Composite: "text-indigo-400",
  "Agent Presets": "text-indigo-400",
  Skill: "text-violet-400",
  Agent: "text-indigo-400",
  Server: "text-teal-400",
  Custom: "text-gray-400",
};

/* ── Node type → icon resolution ── */

const NODE_TYPE_ICONS: Record<string, string> = {
  iterIn: "RefreshCw",
  iterOut: "RefreshCw",
  llmCall: "MessageSquare",
  compositeNode: "Puzzle",
  graphIn: "ArrowDownToLine",
  graphOut: "ArrowUpFromLine",
  imageViewer: "Images",
  textViewer: "FileText",
  textScroll: "ScrollText",
  actionLog: "ListOrdered",
  metrics: "BarChart3",
  stateContainer: "ClipboardList",
  tool: "Wrench",
  policyForward: "Brain",
};

function resolveNodeIcon(nodeType: string): ReactNode {
  // Known built-in type
  const direct = NODE_TYPE_ICONS[nodeType];
  if (direct && ICONS[direct]) return ICONS[direct];

  // Nodeset node (contains __)
  if (nodeType.includes("__")) {
    const prefix = nodeType.split("__")[0];
    if (prefix.startsWith("env"))
      return ICONS["Server"] || <Server size={14} />;
    if (prefix.startsWith("policy"))
      return ICONS["Brain"] || <Brain size={14} />;
    return ICONS["Wrench"] || <Wrench size={14} />;
  }

  return <Puzzle size={14} />;
}

/* ── Drag helper ── */

function onDragStart(e: React.DragEvent, entry: CatalogEntry) {
  e.dataTransfer.setData("application/reactflow", entry.type);
  if (entry.data) {
    e.dataTransfer.setData(
      "application/reactflow-data",
      JSON.stringify(entry.data),
    );
  }
  e.dataTransfer.effectAllowed = "move";
}

/* ── Section header ── */

function SectionHeader({
  name,
  count,
  open,
  onToggle,
  action,
}: {
  name: string;
  count: number;
  open: boolean;
  onToggle: () => void;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-center">
      <button
        onClick={onToggle}
        className={clsx(
          "flex flex-1 items-center gap-1.5 rounded px-2 py-1.5 text-left text-[11px] font-semibold uppercase tracking-wider transition",
          "hover:bg-gray-800/70",
          CATEGORY_COLORS[name] || "text-gray-400",
        )}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {CATEGORY_ICONS[name]}
        <span className="flex-1">{name}</span>
        <span className="text-[10px] font-normal text-gray-600">{count}</span>
      </button>
      {action}
    </div>
  );
}

/* ── Recursive node tree item ── */

function NodeTreeItem({
  node,
  keyPrefix,
  expanded,
  onToggle,
}: {
  node: NodeDef;
  keyPrefix: string;
  expanded: Set<string>;
  onToggle: (key: string) => void;
}) {
  const isComposite = !!node.subgraph;
  const itemKey = `${keyPrefix}:${node.id}`;
  const isOpen = expanded.has(itemKey);

  return (
    <div>
      <div
        className={clsx(
          "flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] text-gray-400 transition",
          isComposite && "hover:bg-gray-800/70 cursor-pointer",
        )}
        onClick={isComposite ? () => onToggle(itemKey) : undefined}
        draggable={isComposite}
        onDragStart={
          isComposite
            ? (e) => {
                e.dataTransfer.setData(
                  "application/reactflow",
                  "compositeNode",
                );
                e.dataTransfer.setData(
                  "application/reactflow-data",
                  JSON.stringify({
                    subgraph: node.subgraph,
                  }),
                );
                e.dataTransfer.effectAllowed = "move";
              }
            : undefined
        }
      >
        {isComposite ? (
          isOpen ? (
            <ChevronDown size={10} className="shrink-0 text-gray-600" />
          ) : (
            <ChevronRight size={10} className="shrink-0 text-gray-600" />
          )
        ) : (
          <span className="w-[10px] shrink-0" />
        )}
        <span className="shrink-0">{resolveNodeIcon(node.type)}</span>
        <span className="truncate" title={`${node.type} — ${node.id}`}>
          {node.label || node.type.replace(/__/g, ": ")}
        </span>
      </div>

      {isComposite && isOpen && node.subgraph && (
        <div className="ml-3 border-l border-gray-800 pl-2">
          {node.subgraph.nodes.map((child) => (
            <NodeTreeItem
              key={child.id}
              node={child}
              keyPrefix={itemKey}
              expanded={expanded}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Shared graph data provider ── */

/** Folder keys to auto-expand on first load: every folder whose leaf name
 *  is "verified", plus all its ancestors (so the verified leaf is actually
 *  reachable). Keyed the same way FolderRow reads — __folder__:graph:<path>. */
function defaultExpandedFolderKeys(graphFolders: string[]): string[] {
  const keys = new Set<string>();
  for (const fp of graphFolders) {
    if ((fp.split("/").pop() || "") !== "verified") continue;
    const parts = fp.split("/");
    for (let i = 1; i <= parts.length; i++) {
      keys.add(`__folder__:graph:${parts.slice(0, i).join("/")}`);
    }
  }
  return [...keys];
}

function useGraphData() {
  const [allGraphs, setAllGraphs] = useState<SavedGraph[]>([]);
  const [folders, setFolders] = useState<{ graph: string[]; node: string[] }>({
    graph: [],
    node: [],
  });
  const [loading, setLoading] = useState(false);
  const [loadStatus, setLoadStatus] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  // Name of the graph currently being opened (drives the loading overlay);
  // null when idle. The AbortController lets the Cancel button drop the
  // in-flight nodeset-activation request.
  const [loadingGraph, setLoadingGraph] = useState<string | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  // Seed the default "all verified folders expanded" view exactly once, on
  // first successful folder load. Guarded so later refreshes / disk-change
  // re-fetches don't fight the user's manual collapses.
  const seededDefaultRef = useRef(false);

  const cancelLoad = useCallback(() => {
    loadAbortRef.current?.abort();
  }, []);

  const toggleExpanded = useCallback((key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [data, graphFolders, nodeFolders] = await Promise.all([
        api.listGraphs(),
        api.listGraphFolders("graph").catch(() => [] as string[]),
        api.listGraphFolders("node").catch(() => [] as string[]),
      ]);
      setAllGraphs(data);
      setFolders({ graph: graphFolders, node: nodeFolders });
      if (!seededDefaultRef.current) {
        seededDefaultRef.current = true;
        const defaults = defaultExpandedFolderKeys(graphFolders);
        if (defaults.length > 0) {
          setExpanded((prev) => {
            const next = new Set(prev);
            for (const k of defaults) next.add(k);
            return next;
          });
        }
      }
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    (window as unknown as Record<string, unknown>).__refreshSavedGraphs =
      refresh;
    return () => {
      delete (window as unknown as Record<string, unknown>)
        .__refreshSavedGraphs;
    };
  }, [refresh]);

  const handleDelete = useCallback(
    async (e: React.MouseEvent, id: string) => {
      e.stopPropagation();
      try {
        await api.deleteGraph(id);
        refresh();
      } catch {
        /* ignore */
      }
    },
    [refresh],
  );

  const handleLoadInActive = useCallback(
    async (e: React.MouseEvent, graph: SavedGraph) => {
      e.stopPropagation();
      const nodeTypes = (graph.nodes as Array<{ type: string }>).map(
        (n) => n.type,
      );
      const controller = new AbortController();
      loadAbortRef.current = controller;
      setLoadingGraph(graph.name);
      setLoadStatus("Loading nodesets...");
      try {
        const result = await ensureNodesetsLoaded(nodeTypes, controller.signal);
        if (result.unknown.length > 0) {
          setLoadStatus(`Unknown nodesets: ${result.unknown.join(", ")}`);
          setTimeout(() => setLoadStatus(null), 4000);
        } else if (result.failed.length > 0) {
          setLoadStatus(`Failed to load: ${result.failed.join(", ")}`);
          setTimeout(() => setLoadStatus(null), 4000);
        } else {
          setLoadStatus(null);
        }
        // Notify CanvasPage to refresh schemas (so subgraph editors get updated schemas)
        if (result.loaded.length > 0) {
          window.dispatchEvent(new Event("nodesets-changed"));
        }
        let schemas: Record<string, unknown>[] = [];
        try {
          schemas = await api.getNodeSchemas();
        } catch {
          /* ignore */
        }
        // The user may have cancelled while schemas were fetching.
        if (controller.signal.aborted) return;
        useFlowStore.getState().loadGraph(
          {
            name: graph.name,
            description: graph.description,
            nodes: graph.nodes as GraphDefinition["nodes"],
            edges: graph.edges as GraphDefinition["edges"],
            containers: (graph.containers ||
              []) as GraphDefinition["containers"],
            access_grants: (graph.access_grants ||
              []) as GraphDefinition["access_grants"],
            step_budget: graph.step_budget,
            eval_graph: graph.eval_graph,
          },
          schemas,
        );
        useFlowStore
          .getState()
          .updateTabMeta(useFlowStore.getState().activeTabId, {
            title: graph.name,
            graphId: graph._id,
            dirty: false,
          });
        if (result.loaded.length > 0) {
          setLoadStatus(`Loaded: ${result.loaded.join(", ")}`);
          setTimeout(() => setLoadStatus(null), 3000);
        }
      } catch (err) {
        // Cancel is expected — leave the active tab untouched, no error banner.
        if ((err as Error)?.name === "AbortError") {
          setLoadStatus("Load cancelled");
          setTimeout(() => setLoadStatus(null), 2000);
        } else {
          throw err;
        }
      } finally {
        setLoadingGraph(null);
        loadAbortRef.current = null;
      }
    },
    [],
  );

  const handleOpenInTab = useCallback(async (graph: SavedGraph) => {
    const nodeTypes = (graph.nodes as Array<{ type: string }>).map(
      (n) => n.type,
    );
    const controller = new AbortController();
    loadAbortRef.current = controller;
    setLoadingGraph(graph.name);
    try {
      const tabResult = await ensureNodesetsLoaded(
        nodeTypes,
        controller.signal,
      );
      if (tabResult.loaded.length > 0) {
        window.dispatchEvent(new Event("nodesets-changed"));
      }
      let schemas: Record<string, unknown>[] = [];
      try {
        schemas = await api.getNodeSchemas();
      } catch {
        /* ignore */
      }
      // The user may have cancelled while schemas were fetching.
      if (controller.signal.aborted) return;
      useFlowStore.getState().openGraphInTab(
        graph._id,
        {
          name: graph.name,
          description: graph.description,
          nodes: graph.nodes as GraphDefinition["nodes"],
          edges: graph.edges as GraphDefinition["edges"],
          containers: (graph.containers || []) as GraphDefinition["containers"],
          access_grants: (graph.access_grants ||
            []) as GraphDefinition["access_grants"],
          step_budget: graph.step_budget,
          eval_graph: graph.eval_graph,
        },
        schemas,
      );
    } catch (err) {
      // Cancel is expected — just don't open the tab. Re-throw real errors.
      if ((err as Error)?.name !== "AbortError") throw err;
    } finally {
      setLoadingGraph(null);
      loadAbortRef.current = null;
    }
  }, []);

  return {
    allGraphs,
    folders,
    loading,
    loadStatus,
    loadingGraph,
    cancelLoad,
    expanded,
    toggleExpanded,
    refresh,
    handleDelete,
    handleLoadInActive,
    handleOpenInTab,
  };
}

/* ── Graph item row (shared by both sections) ── */

function GraphItemRow({
  g,
  expanded,
  toggleExpanded,
  handleDelete,
  handleLoadInActive,
  handleOpenInTab,
}: {
  g: SavedGraph;
  expanded: Set<string>;
  toggleExpanded: (key: string) => void;
  handleDelete: (e: React.MouseEvent, id: string) => void;
  handleLoadInActive: (e: React.MouseEvent, graph: SavedGraph) => void;
  handleOpenInTab: (graph: SavedGraph) => void;
}) {
  const isNode = g.kind === "node";
  const graphKey = `${isNode ? "node" : "graph"}:${g._id}`;
  const isOpen = expanded.has(graphKey);
  const nodes = g.nodes as NodeDef[];

  return (
    <div>
      <div
        draggable
        onDragStart={(e) => {
          // Intra-Explorer move payload — folder drop targets read this.
          e.dataTransfer.setData(
            "application/agentcanvas-graph-move",
            JSON.stringify({ id: g._id, kind: isNode ? "node" : "graph" }),
          );
          // Graph nodes additionally drag onto the canvas as composites.
          if (isNode) {
            e.dataTransfer.setData("application/reactflow", "compositeNode");
            e.dataTransfer.setData(
              "application/reactflow-data",
              JSON.stringify({
                subgraph: {
                  name: g.name,
                  description: g.description,
                  nodes: g.nodes,
                  edges: g.edges,
                  step_budget: g.step_budget,
                },
              }),
            );
          }
          e.dataTransfer.effectAllowed = "move";
        }}
        onDoubleClick={!isNode ? () => handleOpenInTab(g) : undefined}
        className={clsx(
          "group flex items-center gap-1.5 rounded px-2 py-1 text-xs text-gray-300 transition hover:bg-gray-800",
          isNode ? "cursor-grab active:cursor-grabbing" : "cursor-pointer",
        )}
      >
        {nodes.length > 0 ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleExpanded(graphKey);
            }}
            className="shrink-0 rounded p-0.5 text-gray-600 hover:text-gray-400"
            draggable={false}
          >
            {isOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          </button>
        ) : (
          <span className="w-[18px] shrink-0" />
        )}
        {isNode ? (
          <Component size={12} className="shrink-0 text-indigo-400/70" />
        ) : (
          <Workflow size={12} className="shrink-0 text-amber-500/70" />
        )}
        <span className="flex-1 truncate" title={g.description || g.name}>
          {g.name}
        </span>
        <span className="text-[9px] text-gray-600">{nodes.length}</span>
        {!isNode && (
          <button
            onClick={(e) => handleLoadInActive(e, g)}
            className="hidden rounded p-0.5 text-gray-600 hover:bg-gray-700 hover:text-blue-400 group-hover:block"
            title="Load in active tab"
            draggable={false}
          >
            <Upload size={10} />
          </button>
        )}
        <button
          onClick={(e) => handleDelete(e, g._id)}
          className="hidden rounded p-0.5 text-gray-600 hover:bg-gray-700 hover:text-red-400 group-hover:block"
          title="Delete"
          draggable={false}
        >
          <Trash2 size={10} />
        </button>
      </div>

      {isOpen && nodes.length > 0 && (
        <div className="ml-3 border-l border-gray-800 pl-2">
          {nodes.map((node) => (
            <NodeTreeItem
              key={node.id}
              node={node}
              keyPrefix={graphKey}
              expanded={expanded}
              onToggle={toggleExpanded}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Folder tree (real subdirectories under each kind root) ── */

type SharedRowProps = {
  expanded: Set<string>;
  toggleExpanded: (key: string) => void;
  handleDelete: (e: React.MouseEvent, id: string) => void;
  handleLoadInActive: (e: React.MouseEvent, graph: SavedGraph) => void;
  handleOpenInTab: (graph: SavedGraph) => void;
};

interface GraphTreeNode {
  graphs: SavedGraph[];
  folders: Map<string, GraphTreeNode>;
}

interface FolderCtx {
  kind: "graph" | "node";
  dropTargetKey: string | null;
  setDropTargetKey: (k: string | null) => void;
  onDropGraph: (e: React.DragEvent, destFolder: string) => void;
  onNewFolder: (parent: string) => void;
  onRenameFolder: (path: string) => void;
  onDeleteFolder: (path: string, count: number) => void;
}

/** Build a nested folder tree from each graph's `folder` path plus the
 *  explicit folder list (the latter surfaces empty folders too). */
function buildGraphTree(
  graphs: SavedGraph[],
  folderPaths: string[],
): GraphTreeNode {
  const root: GraphTreeNode = { graphs: [], folders: new Map() };
  const ensure = (path: string): GraphTreeNode => {
    let cur = root;
    if (!path) return cur;
    for (const part of path.split("/")) {
      if (!part) continue;
      let next = cur.folders.get(part);
      if (!next) {
        next = { graphs: [], folders: new Map() };
        cur.folders.set(part, next);
      }
      cur = next;
    }
    return cur;
  };
  for (const fp of folderPaths) ensure(fp);
  for (const g of graphs) ensure(g.folder || "").graphs.push(g);
  return root;
}

function countTreeItems(node: GraphTreeNode): number {
  let n = node.graphs.length;
  for (const child of node.folders.values()) n += countTreeItems(child);
  return n;
}

function hasGraphMovePayload(e: React.DragEvent): boolean {
  return Array.from(e.dataTransfer.types).includes(
    "application/agentcanvas-graph-move",
  );
}

function FolderTreeView({
  node,
  path,
  ctx,
  sharedProps,
}: {
  node: GraphTreeNode;
  path: string;
  ctx: FolderCtx;
  sharedProps: SharedRowProps;
}) {
  const folderNames = [...node.folders.keys()].sort((a, b) =>
    a.localeCompare(b),
  );
  const graphs = [...node.graphs].sort((a, b) => a.name.localeCompare(b.name));
  return (
    <>
      {folderNames.map((fn) => (
        <FolderRow
          key={`folder:${fn}`}
          name={fn}
          path={path ? `${path}/${fn}` : fn}
          node={node.folders.get(fn)!}
          ctx={ctx}
          sharedProps={sharedProps}
        />
      ))}
      {graphs.map((g) => (
        <GraphItemRow key={g._id} g={g} {...sharedProps} />
      ))}
    </>
  );
}

function FolderRow({
  name,
  path,
  node,
  ctx,
  sharedProps,
}: {
  name: string;
  path: string;
  node: GraphTreeNode;
  ctx: FolderCtx;
  sharedProps: SharedRowProps;
}) {
  const folderKey = `__folder__:${ctx.kind}:${path}`;
  const isOpen = sharedProps.expanded.has(folderKey);
  const count = countTreeItems(node);
  const isDropTarget = ctx.dropTargetKey === folderKey;

  // DnD lives on the outer wrapper (stopPropagation) so the deepest folder
  // under the cursor claims the drop — dropping anywhere inside folder "a"
  // (including on a child graph row) moves the item into "a".
  return (
    <div
      onDragOver={(e) => {
        if (!hasGraphMovePayload(e)) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "move";
        ctx.setDropTargetKey(folderKey);
      }}
      onDragLeave={(e) => {
        e.stopPropagation();
        if (ctx.dropTargetKey === folderKey) ctx.setDropTargetKey(null);
      }}
      onDrop={(e) => {
        e.stopPropagation();
        ctx.onDropGraph(e, path);
      }}
    >
      <div
        onClick={() => sharedProps.toggleExpanded(folderKey)}
        className={clsx(
          "group flex cursor-pointer items-center gap-1.5 rounded px-2 py-1 text-xs text-amber-300/90 transition hover:bg-gray-800",
          isDropTarget && "bg-blue-500/15 ring-1 ring-blue-500/60",
        )}
      >
        {isOpen ? (
          <ChevronDown size={10} className="shrink-0 text-gray-600" />
        ) : (
          <ChevronRight size={10} className="shrink-0 text-gray-600" />
        )}
        {isOpen ? (
          <FolderOpen size={12} className="shrink-0 text-amber-500/80" />
        ) : (
          <Folder size={12} className="shrink-0 text-amber-500/80" />
        )}
        <span className="flex-1 truncate" title={path}>
          {name}
        </span>
        <span className="text-[9px] text-gray-600">{count}</span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            ctx.onNewFolder(path);
          }}
          className="hidden rounded p-0.5 text-gray-600 hover:bg-gray-700 hover:text-amber-400 group-hover:block"
          title="New subfolder"
          draggable={false}
        >
          <FolderPlus size={10} />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            ctx.onRenameFolder(path);
          }}
          className="hidden rounded p-0.5 text-gray-600 hover:bg-gray-700 hover:text-blue-400 group-hover:block"
          title="Rename folder"
          draggable={false}
        >
          <FileEdit size={10} />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            ctx.onDeleteFolder(path, count);
          }}
          className="hidden rounded p-0.5 text-gray-600 hover:bg-gray-700 hover:text-red-400 group-hover:block"
          title="Delete folder"
          draggable={false}
        >
          <Trash2 size={10} />
        </button>
      </div>
      {isOpen && (
        <div className="ml-3 border-l border-gray-800 pl-2">
          <FolderTreeView
            node={node}
            path={path}
            ctx={ctx}
            sharedProps={sharedProps}
          />
        </div>
      )}
    </div>
  );
}

/* ── Category block 1: Graphs  +  Category block 2: Graph Nodes ── */

function GraphsSections() {
  const {
    allGraphs,
    folders,
    loading,
    loadStatus,
    loadingGraph,
    cancelLoad,
    expanded,
    toggleExpanded,
    refresh,
    handleDelete,
    handleLoadInActive,
    handleOpenInTab,
  } = useGraphData();

  const [graphsOpen, setGraphsOpen] = useState(true);
  const [nodesOpen, setNodesOpen] = useState(true);
  const [dropTargetKey, setDropTargetKey] = useState<string | null>(null);

  const graphs = allGraphs.filter((g) => g.kind !== "node");
  const graphNodes = allGraphs.filter((g) => g.kind === "node");
  const graphTree = buildGraphTree(graphs, folders.graph);
  const nodeTree = buildGraphTree(graphNodes, folders.node);

  const sharedProps: SharedRowProps = {
    expanded,
    toggleExpanded,
    handleDelete,
    handleLoadInActive,
    handleOpenInTab,
  };

  const handleMove = useCallback(
    async (id: string, destFolder: string) => {
      try {
        await api.moveGraph(id, destFolder);
      } catch {
        return;
      }
      refresh();
    },
    [refresh],
  );

  const makeCtx = (kind: "graph" | "node"): FolderCtx => ({
    kind,
    dropTargetKey,
    setDropTargetKey,
    onDropGraph: (e, destFolder) => {
      e.preventDefault();
      setDropTargetKey(null);
      const raw = e.dataTransfer.getData("application/agentcanvas-graph-move");
      if (!raw) return;
      let payload: { id: string; kind: string };
      try {
        payload = JSON.parse(raw);
      } catch {
        return;
      }
      // Don't let a graph cross into the graph_nodes tree or vice-versa.
      if (payload.kind !== kind) return;
      handleMove(payload.id, destFolder);
    },
    onNewFolder: (parent) => {
      const name = window.prompt("New folder name");
      if (!name || !name.trim()) return;
      const path = parent ? `${parent}/${name.trim()}` : name.trim();
      api
        .createGraphFolder(kind, path)
        .then(refresh)
        .catch(() => {});
    },
    onRenameFolder: (path) => {
      const cur = path.split("/").pop() || path;
      const name = window.prompt("Rename folder", cur);
      if (!name || !name.trim() || name.trim() === cur) return;
      const parent = path.split("/").slice(0, -1).join("/");
      const newPath = parent ? `${parent}/${name.trim()}` : name.trim();
      api
        .renameGraphFolder(kind, path, newPath)
        .then(refresh)
        .catch(() => {});
    },
    onDeleteFolder: (path, count) => {
      const msg =
        count > 0
          ? `Delete folder "${path}" and ${count} item(s) inside?`
          : `Delete empty folder "${path}"?`;
      if (!window.confirm(msg)) return;
      api
        .deleteGraphFolder(kind, path, count > 0)
        .then(refresh)
        .catch(() => {});
    },
  });

  const graphCtx = makeCtx("graph");
  const nodeCtx = makeCtx("node");

  // Root drop zone (destFolder = "") for each section.
  const rootDropProps = (ctx: FolderCtx) => {
    const rootKey = `__folder__:${ctx.kind}:`;
    return {
      onDragOver: (e: React.DragEvent) => {
        if (!hasGraphMovePayload(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        setDropTargetKey(rootKey);
      },
      onDragLeave: () => {
        if (dropTargetKey === rootKey) setDropTargetKey(null);
      },
      onDrop: (e: React.DragEvent) => ctx.onDropGraph(e, ""),
      className: clsx(
        "ml-3 space-y-0.5 border-l border-gray-800 pl-2 pt-0.5",
        dropTargetKey === rootKey && "bg-blue-500/10",
      ),
    };
  };

  return (
    <>
      {/* ── Category block 1: Graphs ── */}
      <div className="mb-1">
        <SectionHeader
          name="Graphs"
          count={loading ? 0 : graphs.length}
          open={graphsOpen}
          onToggle={() => setGraphsOpen(!graphsOpen)}
          action={
            <div className="flex items-center">
              <button
                onClick={() => graphCtx.onNewFolder("")}
                className="rounded p-1 text-gray-600 hover:bg-gray-800 hover:text-amber-400"
                title="New folder"
              >
                <FolderPlus size={10} />
              </button>
              <button
                onClick={refresh}
                className="mr-1 rounded p-1 text-gray-600 hover:bg-gray-800 hover:text-gray-400"
                title="Refresh"
              >
                <RotateCw size={10} className={loading ? "animate-spin" : ""} />
              </button>
            </div>
          }
        />
        {graphsOpen && (
          <div {...rootDropProps(graphCtx)}>
            {graphs.length === 0 &&
              graphTree.folders.size === 0 &&
              !loading && (
                <div className="px-2 py-1 text-[10px] text-gray-600">
                  No saved graphs
                </div>
              )}
            <FolderTreeView
              node={graphTree}
              path=""
              ctx={graphCtx}
              sharedProps={sharedProps}
            />
          </div>
        )}
      </div>

      {/* ── Divider 1 ── */}
      <div className="mx-2 my-1 border-t border-gray-800" />

      {/* ── Category block 2: Graph Nodes ── */}
      <div className="mb-1">
        <SectionHeader
          name="Graph Nodes"
          count={loading ? 0 : graphNodes.length}
          open={nodesOpen}
          onToggle={() => setNodesOpen(!nodesOpen)}
          action={
            <button
              onClick={() => nodeCtx.onNewFolder("")}
              className="mr-1 rounded p-1 text-gray-600 hover:bg-gray-800 hover:text-indigo-400"
              title="New folder"
            >
              <FolderPlus size={10} />
            </button>
          }
        />
        {nodesOpen && (
          <div {...rootDropProps(nodeCtx)}>
            {graphNodes.length === 0 &&
              nodeTree.folders.size === 0 &&
              !loading && (
                <div className="px-2 py-1 text-[10px] text-gray-600">
                  No graph nodes — use &quot;Save as Node&quot;
                </div>
              )}
            <FolderTreeView
              node={nodeTree}
              path=""
              ctx={nodeCtx}
              sharedProps={sharedProps}
            />
          </div>
        )}
      </div>

      {/* Status message */}
      {loadStatus && (
        <div className="mx-3 mt-1 px-2 py-1 text-[10px] text-yellow-400">
          {loadStatus}
        </div>
      )}

      {/* Loading overlay — shown while a graph's nodesets are being
          activated. Cancel aborts the in-flight request and skips the open. */}
      {loadingGraph && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-[1px]">
          <div className="flex w-[280px] flex-col items-center gap-3 rounded-lg border border-gray-700 bg-gray-900 px-6 py-5 shadow-xl">
            <Loader2 size={28} className="animate-spin text-blue-400" />
            <div className="text-sm font-medium text-gray-200">
              Loading graph…
            </div>
            <div
              className="max-w-full truncate text-xs text-gray-400"
              title={loadingGraph}
            >
              {loadingGraph}
            </div>
            <div className="text-[10px] text-gray-500">
              Activating required nodesets
            </div>
            <button
              onClick={cancelLoad}
              className="mt-1 rounded bg-gray-800 px-3 py-1 text-xs text-gray-300 transition hover:bg-gray-700 hover:text-red-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}

/* ── Nodes catalog section ── */

function CategoryGroup({
  name,
  items,
  defaultOpen = false,
  forceOpen = false,
}: {
  name: string;
  items: CatalogEntry[];
  defaultOpen?: boolean;
  forceOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const effectiveOpen = forceOpen || open;

  return (
    <div className="mb-0.5">
      <button
        onClick={() => setOpen(!open)}
        className={clsx(
          "flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-left text-xs font-medium transition",
          "hover:bg-gray-800/70",
          CATEGORY_COLORS[name] || "text-gray-400",
        )}
      >
        {effectiveOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {CATEGORY_ICONS[name]}
        <span className="flex-1">{name}</span>
        <span className="text-[10px] font-normal text-gray-600">
          {items.length}
        </span>
      </button>
      {effectiveOpen && (
        <div className="ml-3 space-y-0.5 border-l border-gray-800 pl-2 pt-0.5">
          {items.map((item, i) => (
            <div
              key={`${item.type}-${item.label}-${i}`}
              draggable
              onDragStart={(e) => onDragStart(e, item)}
              className="flex cursor-grab items-center gap-2 rounded px-2 py-1 text-xs text-gray-300 transition hover:bg-gray-800 active:cursor-grabbing"
            >
              {ICONS[item.icon] || <Wrench size={14} />}
              {item.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Main component ── */

interface ExplorerPanelProps {
  catalog: CatalogEntry[];
}

const WIDTH_KEY = "agentcanvas.explorerWidth";
const MIN_WIDTH = 140;
const DEFAULT_WIDTH = 200;
const MAX_WIDTH_PAD = 200; // leave at least this much for the canvas

function readStoredWidth(): number {
  try {
    const raw = localStorage.getItem(WIDTH_KEY);
    if (!raw) return DEFAULT_WIDTH;
    const n = Number(raw);
    return Number.isFinite(n) && n >= MIN_WIDTH ? n : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH;
  }
}

function clampWidth(w: number): number {
  const max = Math.max(MIN_WIDTH, window.innerWidth - MAX_WIDTH_PAD);
  return Math.min(max, Math.max(MIN_WIDTH, w));
}

export default function ExplorerPanel({ catalog }: ExplorerPanelProps) {
  const [nodesOpen, setNodesOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [width, setWidth] = useState<number>(() =>
    clampWidth(readStoredWidth()),
  );
  const [dragging, setDragging] = useState(false);
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(
    null,
  );

  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      /* ignore quota / privacy-mode errors */
    }
  }, [width]);

  useEffect(() => {
    const onResize = () => setWidth((w) => clampWidth(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      dragStateRef.current = { startX: e.clientX, startWidth: width };
      setDragging(true);
    },
    [width],
  );

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    setWidth(clampWidth(drag.startWidth + dx));
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStateRef.current) return;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    dragStateRef.current = null;
    setDragging(false);
  }, []);

  const trimmed = query.trim().toLowerCase();
  const filtered = trimmed
    ? catalog.filter((c) =>
        [c.label, c.type, c.category]
          .filter(Boolean)
          .some((s) => s.toLowerCase().includes(trimmed)),
      )
    : catalog;
  const categories = [...new Set(filtered.map((c) => c.category))];

  return (
    <div
      className="relative flex h-full flex-col border-r border-gray-800 bg-gray-900"
      style={{ width: `${width}px`, flexShrink: 0 }}
    >
      <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
        Explorer
      </div>
      <div className="border-b border-gray-800 px-2 py-1.5">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search nodes…"
          className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1 text-[11px] text-gray-200 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none"
        />
      </div>
      <div className="flex-1 overflow-auto py-1">
        {/* Category blocks 1 + 2: Graphs, divider, Graph Nodes */}
        {!trimmed && <GraphsSections />}

        {/* Divider 2 */}
        {!trimmed && <div className="mx-2 my-1 border-t border-gray-800" />}

        {/* Category block 3: Fundamental nodes */}
        <SectionHeader
          name="Nodes"
          count={filtered.length}
          open={trimmed ? true : nodesOpen}
          onToggle={() => setNodesOpen(!nodesOpen)}
        />
        {(trimmed ? true : nodesOpen) && (
          <div className="ml-3 border-l border-gray-800 pl-2 pt-0.5">
            {categories.length === 0 && trimmed ? (
              <div className="px-2 py-2 text-[11px] text-gray-600">
                No matches.
              </div>
            ) : (
              categories.map((cat) => (
                <CategoryGroup
                  key={cat}
                  name={cat}
                  items={filtered.filter((c) => c.category === cat)}
                  defaultOpen={["Environment", "LLM", "Control"].includes(cat)}
                  forceOpen={Boolean(trimmed)}
                />
              ))
            )}
          </div>
        )}
      </div>
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize explorer panel"
        data-dragging={dragging || undefined}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        title="Drag to resize · double-click to reset"
        className="resize-handle-vertical absolute right-0 top-0 h-full"
      />
    </div>
  );
}
