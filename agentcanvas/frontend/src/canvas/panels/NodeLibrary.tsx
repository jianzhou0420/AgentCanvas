/** Sidebar: unified node catalog — all node types available at every canvas level. */

import { useState } from "react";
import {
  Bot,
  Wrench,
  Puzzle,
  Globe,
  Image,
  Compass,
  Database,
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
} from "lucide-react";
import clsx from "clsx";
import type { ReactNode } from "react";
import type { CatalogEntry } from "../types";

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
  Database: <Database size={14} />,
};

const CATEGORY_ICONS: Record<string, ReactNode> = {
  Environment: <Server size={12} />,
  Policy: <Brain size={12} />,
  LLM: <MessageSquare size={12} />,
  Prompt: <FileEdit size={12} />,
  History: <RefreshCw size={12} />,
  Tool: <Wrench size={12} />,
  Decision: <GitBranch size={12} />,
  "Loop Control": <CircleStop size={12} />,
  Output: <Eye size={12} />,
  Composite: <Puzzle size={12} />,
  Boundary: <ArrowDownToLine size={12} />,
  "Agent Presets": <Bot size={12} />,
  State: <Database size={12} />,
  "Saved Graphs": <FolderOpen size={12} />,
  Skill: <Puzzle size={12} />,
  Agent: <Bot size={12} />,
  Server: <Globe size={12} />,
  Custom: <Wrench size={12} />,
};

const CATEGORY_COLORS: Record<string, string> = {
  Environment: "text-green-400",
  Policy: "text-blue-400",
  LLM: "text-yellow-400",
  Prompt: "text-purple-400",
  History: "text-emerald-400",
  Tool: "text-cyan-400",
  Decision: "text-pink-400",
  "Loop Control": "text-red-400",
  Output: "text-orange-400",
  Composite: "text-indigo-400",
  Boundary: "text-gray-400",
  State: "text-violet-400",
  "Agent Presets": "text-indigo-400",
  "Saved Graphs": "text-amber-400",
  Skill: "text-violet-400",
  Agent: "text-indigo-400",
  Server: "text-teal-400",
  Custom: "text-gray-400",
};

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

interface CategoryGroupProps {
  name: string;
  items: CatalogEntry[];
  defaultOpen?: boolean;
}

function CategoryGroup({
  name,
  items,
  defaultOpen = false,
}: CategoryGroupProps) {
  const [open, setOpen] = useState(defaultOpen);

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
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {CATEGORY_ICONS[name]}
        <span className="flex-1">{name}</span>
        <span className="text-[10px] font-normal text-gray-600">
          {items.length}
        </span>
      </button>
      {open && (
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

/* ── Saved Graphs section ── */

import { useEffect, useCallback } from "react";
import { Trash2, Upload } from "lucide-react";
import { api } from "../../api";
import { useFlowStore } from "../useFlowStore";
import type { SavedGraph } from "../../types";
import type { GraphDefinition } from "../types";

function SavedGraphsSection() {
  const [graphs, setGraphs] = useState<SavedGraph[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listGraphs();
      setGraphs(data);
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Expose refresh globally so SaveGraphDialog can trigger it
  useEffect(() => {
    (window as unknown as Record<string, unknown>).__refreshSavedGraphs =
      refresh;
    return () => {
      delete (window as unknown as Record<string, unknown>)
        .__refreshSavedGraphs;
    };
  }, [refresh]);

  const handleDelete = async (id: string) => {
    try {
      await api.deleteGraph(id);
      refresh();
    } catch {
      /* ignore */
    }
  };

  const handleLoadAsRoot = (graph: SavedGraph) => {
    useFlowStore.getState().loadGraph({
      name: graph.name,
      description: graph.description,
      nodes: graph.nodes as GraphDefinition["nodes"],
      edges: graph.edges as GraphDefinition["edges"],
      containers: (graph.containers || []) as GraphDefinition["containers"],
      access_grants: (graph.access_grants ||
        []) as GraphDefinition["access_grants"],
      step_budget: graph.step_budget,
    });
  };

  if (graphs.length === 0 && !loading) return null;

  return (
    <div className="mb-0.5">
      <div
        className={clsx(
          "flex w-full items-center gap-1.5 rounded px-2 py-1.5 text-left text-xs font-medium",
          CATEGORY_COLORS["Saved Graphs"] || "text-amber-400",
        )}
      >
        {CATEGORY_ICONS["Saved Graphs"]}
        <span className="flex-1">Saved Graphs</span>
        <span className="text-[10px] font-normal text-gray-600">
          {loading ? "..." : graphs.length}
        </span>
      </div>
      <div className="ml-3 space-y-0.5 border-l border-gray-800 pl-2 pt-0.5">
        {graphs.map((g) => (
          <div
            key={g._id}
            draggable
            onDragStart={(e) => {
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
              e.dataTransfer.effectAllowed = "move";
            }}
            className="group flex cursor-grab items-center gap-1.5 rounded px-2 py-1 text-xs text-gray-300 transition hover:bg-gray-800 active:cursor-grabbing"
          >
            <FolderOpen size={12} className="shrink-0 text-amber-500" />
            <span className="flex-1 truncate">{g.name}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleLoadAsRoot(g);
              }}
              className="hidden rounded p-0.5 text-gray-500 hover:bg-gray-700 hover:text-blue-400 group-hover:block"
              title="Load as root graph"
            >
              <Upload size={10} />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(g._id);
              }}
              className="hidden rounded p-0.5 text-gray-500 hover:bg-gray-700 hover:text-red-400 group-hover:block"
              title="Delete"
            >
              <Trash2 size={10} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main component ── */

interface NodeLibraryProps {
  catalog: CatalogEntry[];
}

export default function NodeLibrary({ catalog }: NodeLibraryProps) {
  const categories = [...new Set(catalog.map((c) => c.category))];

  return (
    <div className="flex h-full w-[170px] flex-col border-r border-gray-800 bg-gray-900">
      <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
        Nodes
      </div>
      <div className="flex-1 overflow-auto py-1">
        {categories.map((cat) => (
          <CategoryGroup
            key={cat}
            name={cat}
            items={catalog.filter((c) => c.category === cat)}
            defaultOpen={["Environment", "LLM", "Loop Control"].includes(cat)}
          />
        ))}

        {/* Saved graphs from workspace/graphs/ */}
        <div className="mx-2 my-1 border-t border-gray-800" />
        <SavedGraphsSection />
      </div>
    </div>
  );
}
