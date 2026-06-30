/** Unified node catalog — single catalog for all canvas levels.
 *
 * Built-in node entries are derived from backend /node-schemas (Python-driven).
 * Nodeset nodes + saved graph presets are added dynamically from the backend.
 * Falls back to STATIC_FALLBACK if schemas haven't loaded yet.
 */

import type { CatalogEntry } from "./types";

// ── NodeSchema type (mirrors GenericBlockRenderer's NodeSchema) ──

export interface NodeSchema {
  type: string;
  display_name?: string;
  description?: string;
  category?: string;
  icon?: string;
  kind?: string;
  config_schema?: Record<string, unknown>;
  default_config?: Record<string, unknown>;
  ui_config?: Record<string, unknown>;
  input_ports?: unknown[];
  output_ports?: unknown[];
}

// ── Static fallback — used when backend schemas haven't loaded yet ──
// Keeps the catalog usable on first render before the API responds.

export const UNIFIED_STATIC_CATALOG: CatalogEntry[] = [
  {
    type: "llmCall",
    label: "LLM Call",
    icon: "MessageSquare",
    category: "LLM",
    data: { template: "" },
  },
  {
    type: "iterIn",
    label: "Iter In",
    icon: "RefreshCw",
    category: "Control",
    data: { paired: true },
  },
  {
    type: "imageViewer",
    label: "Image Viewer",
    icon: "Images",
    category: "Output",
    data: {
      rows: 1,
      cols: 2,
      ports: [
        { name: "rgb", wire_type: "IMAGE" },
        { name: "depth", wire_type: "DEPTH" },
      ],
    },
  },
  {
    type: "textViewer",
    label: "Text",
    icon: "FileText",
    category: "Output",
  },
  {
    type: "textScroll",
    label: "Scroll Text",
    icon: "ScrollText",
    category: "Output",
  },
  {
    type: "actionLog",
    label: "Action Log",
    icon: "ListOrdered",
    category: "Output",
  },
  { type: "metrics", label: "Metrics", icon: "BarChart3", category: "Output" },
  {
    type: "stateContainer",
    label: "State Container",
    icon: "Database",
    category: "State",
    data: { label: "State", states: {} },
  },
  {
    type: "compositeNode",
    label: "Empty Composite",
    icon: "Puzzle",
    category: "Composite",
    data: {
      subgraph: {
        name: "New Composite",
        description: "",
        nodes: [],
        edges: [],
      },
    },
  },
  {
    type: "graphIn",
    label: "Graph In",
    icon: "ArrowDownToLine",
    category: "Control",
    data: { portName: "input", wireType: "ANY" },
  },
  {
    type: "graphOut",
    label: "Graph Out",
    icon: "ArrowUpFromLine",
    category: "Control",
    data: { portName: "output", wireType: "ANY" },
  },
];

/** Category display order for the sidebar. */
export const UNIFIED_CATEGORY_ORDER = [
  "Environment",
  "Policy",
  "LLM",
  "History",
  "Tool",
  "Decision",
  "Control",
  "Output",
  "State",
  "Composite",
  "Saved Graphs",
  "Skill",
  "Agent",
  "Server",
  "Custom",
];

/** Acronym categories that must not be title-cased ("llm" → "LLM", not "Llm").
 * Keeping "LLM" exact also preserves the default-open match in ExplorerPanel. */
const CATEGORY_LABELS: Record<string, string> = {
  llm: "LLM",
};

function capitalize(s: string): string {
  return CATEGORY_LABELS[s] || s.charAt(0).toUpperCase() + s.slice(1);
}

/** Build catalog entries for built-in node types from backend schemas.
 * Filters out nodeset nodes (type contains '__'). */
export function buildBuiltinEntries(schemas: NodeSchema[]): CatalogEntry[] {
  const builtins = schemas.filter((s) => !s.type.includes("__"));
  return builtins.map((s) => ({
    type: s.type,
    label: s.display_name || s.type,
    icon: s.icon || "Puzzle",
    category: capitalize(s.category || "custom"),
    data: { ...(s.default_config || {}), _schema: s },
  }));
}

/** Build catalog entries for loaded nodeset nodes (type contains '__'). */
export function buildNodesetEntries(schemas: NodeSchema[]): CatalogEntry[] {
  const nodesetNodes = schemas.filter((s) => s.type.includes("__"));
  return nodesetNodes.map((s) => ({
    type: s.type,
    label: s.display_name || s.type.split("__").pop() || s.type,
    icon: s.icon || "Wrench",
    category: capitalize(s.category || "tool"),
    data: { ...(s.default_config || {}), _schema: s },
  }));
}

/** Build the full unified catalog from backend node schemas.
 * Falls back to static-only if backend schemas unavailable. */
export function buildUnifiedCatalog(
  backendSchemas: NodeSchema[] | null,
): CatalogEntry[] {
  if (backendSchemas && backendSchemas.length > 0) {
    const builtins = buildBuiltinEntries(backendSchemas);
    const nodesetNodes = buildNodesetEntries(backendSchemas);
    return [...builtins, ...nodesetNodes];
  }
  // Fallback: schemas not yet loaded
  return [...UNIFIED_STATIC_CATALOG];
}
