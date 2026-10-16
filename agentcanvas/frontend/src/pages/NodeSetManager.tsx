/** NodeSet Manager page — two-column layout: Resources | Active.
 *
 * Left column (Resources): all discovered nodesets, grouped by
 *   role category (env / method / model / policy / common / other) into
 *   collapsible sections. Unloaded items get a "Load" / "Start" button.
 *
 * Right column (Active): only currently loaded/connected items, grouped by the
 *   same categories, with tool tags and Unload/Stop/Restart controls.
 *
 * A search box in the header filters both columns by name / description / tool.
 *
 * See ADR-008. Category = the workspace/nodesets/<role>/ folder, surfaced by
 * the backend on each NodeSetInfo (see .claude/standard/nodeset-layout.md).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  RefreshCw,
  Loader2,
  Package,
  Search,
  ChevronRight,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import type { NodeSetInfo } from "../types";
import NodeSetCard from "./manager/NodeSetCard";

const noop = async () => {};

// Role buckets (workspace/nodesets/<role>/) surfaced as categories, in display
// order. Mirrors .claude/standard/nodeset-layout.md. Unknown categories are
// appended after these, alphabetically.
const CATEGORY_LABELS: Record<string, string> = {
  env: "Environments",
  method: "Methods",
  model: "Models",
  policy: "Policies",
  common: "Common",
  other: "Other",
};
const CATEGORY_ORDER = ["env", "method", "model", "policy", "common", "other"];

function catLabel(key: string): string {
  return CATEGORY_LABELS[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
}

function matchesQuery(ns: NodeSetInfo, q: string): boolean {
  if (!q) return true;
  const s = q.toLowerCase();
  return (
    ns.name.toLowerCase().includes(s) ||
    (ns.description ?? "").toLowerCase().includes(s) ||
    ns.tools.some((t) => t.toLowerCase().includes(s))
  );
}

interface CategoryBucket {
  key: string;
  label: string;
  items: NodeSetInfo[];
}

function groupByCategory(list: NodeSetInfo[]): CategoryBucket[] {
  const map = new Map<string, NodeSetInfo[]>();
  for (const ns of list) {
    const key = ns.category ?? "other";
    const arr = map.get(key);
    if (arr) arr.push(ns);
    else map.set(key, [ns]);
  }
  const keys = [...map.keys()].sort((a, b) => {
    const ia = CATEGORY_ORDER.indexOf(a);
    const ib = CATEGORY_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
  return keys.map((key) => ({
    key,
    label: catLabel(key),
    items: map
      .get(key)!
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name)),
  }));
}

/** Collapsible section with a chevron, title and count. `forceOpen` overrides
 * the local toggle (used while searching so matches are always visible). */
function CategoryGroup({
  label,
  count,
  forceOpen,
  children,
}: {
  label: string;
  count: number;
  forceOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  const isOpen = forceOpen || open;
  return (
    <section>
      <button
        onClick={() => setOpen((o) => !o)}
        className="mb-2 flex w-full items-center gap-2 text-left"
      >
        <ChevronRight
          size={18}
          className={clsx(
            "text-gray-400 transition-transform",
            isOpen && "rotate-90",
          )}
        />
        <span className="text-base font-semibold uppercase tracking-wide text-gray-200">
          {label}
        </span>
        <span className="rounded bg-gray-800 px-1.5 py-0.5 text-xs text-gray-400">
          {count}
        </span>
      </button>
      {isOpen && <div className="space-y-2">{children}</div>}
    </section>
  );
}

export default function NodeSetManager() {
  const [nodesets, setNodesets] = useState<NodeSetInfo[]>([]);
  const [rescanning, setRescanning] = useState(false);
  const [query, setQuery] = useState("");

  // ── Fetch helpers ──

  const fetchNodesets = useCallback(async () => {
    try {
      setNodesets(await api.listNodesets());
    } catch {
      /* ignore */
    }
  }, []);

  // ── Initial load ──

  useEffect(() => {
    fetchNodesets();
  }, [fetchNodesets]);

  // ── Rescan ──

  const handleRescan = async () => {
    setRescanning(true);
    try {
      await api.reloadComponents();
      await fetchNodesets();
    } finally {
      setRescanning(false);
    }
  };

  // ── NodeSet actions ──

  const handleLoadNodeset = (name: string) => async () => {
    await api.loadNodeset(name);
    await fetchNodesets();
  };

  const handleUnloadNodeset = (name: string) => async () => {
    await api.unloadNodeset(name);
    await fetchNodesets();
  };

  // ── Derived, query-filtered lists ──

  const q = query.trim();

  const visibleNodesets = useMemo(
    () => nodesets.filter((ns) => matchesQuery(ns, q)),
    [nodesets, q],
  );
  const resourceGroups = useMemo(
    () => groupByCategory(visibleNodesets),
    [visibleNodesets],
  );
  const activeGroups = useMemo(
    () => groupByCategory(visibleNodesets.filter((ns) => ns.loaded)),
    [visibleNodesets],
  );

  const hasResources = resourceGroups.length > 0;
  const hasActive = activeGroups.length > 0;

  return (
    <div className="flex h-full w-full flex-col bg-gray-950">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-gray-800 px-6 py-3">
        <div>
          <h1 className="text-xl font-bold text-gray-100">NodeSet Manager</h1>
          <p className="text-sm text-gray-500">
            Manage local and server-mode NodeSets.
          </p>
        </div>
        <button
          onClick={handleRescan}
          disabled={rescanning}
          className="flex items-center gap-2 rounded bg-gray-800 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 disabled:opacity-50"
        >
          {rescanning ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <RefreshCw size={14} />
          )}
          Rescan workspace/
        </button>
      </div>

      {/* Two-column body */}
      <div className="flex flex-1 min-h-0">
        {/* ── Left: Resources ── */}
        <div className="flex w-1/2 flex-col border-r border-gray-800">
          <div className="border-b border-gray-800 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Resources
          </div>
          {/* Search — left-aligned under the Resources tab; filters both columns */}
          <div className="border-b border-gray-800 px-4 py-2">
            <div className="relative w-full max-w-xs">
              <Search
                size={14}
                className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500"
              />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search nodesets…"
                className="w-full rounded border border-gray-800 bg-gray-900 py-1.5 pl-8 pr-3 text-sm text-gray-200 placeholder-gray-600 focus:border-gray-600 focus:outline-none"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!hasResources && (
              <div className="rounded border border-gray-800 bg-gray-900/50 p-3 text-sm text-gray-600">
                {q
                  ? `No nodesets match "${q}"`
                  : "No nodesets discovered"}
              </div>
            )}

            {resourceGroups.map((g) => (
              <CategoryGroup
                key={g.key}
                label={g.label}
                count={g.items.length}
                forceOpen={!!q}
              >
                {g.items.map((ns) => (
                  <NodeSetCard
                    key={ns.name}
                    variant="resource"
                    name={ns.name}
                    description={ns.description}
                    loaded={ns.loaded}
                    tools={ns.tools}
                    requiresServer={ns.requires_server}
                    onLoad={handleLoadNodeset(ns.name)}
                    onUnload={noop}
                  />
                ))}
              </CategoryGroup>
            ))}

          </div>
        </div>

        {/* ── Right: Active ── */}
        <div className="flex w-1/2 flex-col">
          <div className="border-b border-gray-800 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Active
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!hasActive ? (
              <div className="flex h-full items-center justify-center">
                <div className="text-center">
                  <Package size={32} className="mx-auto mb-2 text-gray-700" />
                  <div className="text-sm text-gray-500">
                    {q ? "No active resources match" : "No active resources"}
                  </div>
                  <div className="mt-1 text-xs text-gray-600">
                    Load a nodeset from Resources
                  </div>
                </div>
              </div>
            ) : (
              <>
                {activeGroups.map((g) => (
                  <CategoryGroup
                    key={g.key}
                    label={g.label}
                    count={g.items.length}
                    forceOpen={!!q}
                  >
                    {g.items.map((ns) => (
                      <NodeSetCard
                        key={ns.name}
                        variant="active"
                        name={ns.name}
                        description={ns.description}
                        loaded={ns.loaded}
                        tools={ns.tools}
                        requiresServer={ns.requires_server}
                        onLoad={noop}
                        onUnload={handleUnloadNodeset(ns.name)}
                      />
                    ))}
                  </CategoryGroup>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
