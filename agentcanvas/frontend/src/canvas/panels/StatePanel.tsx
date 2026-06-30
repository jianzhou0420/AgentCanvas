/** State panel — shows all state containers in the bottom output drawer.
 *
 * The editable ``graph_state`` container (home) plus, during a run, every
 * other live container — home (executor-process) and nodeset-owned
 * (subprocess-local) — each tagged by **Owner** ("home" or the nodeset name).
 * Live values arrive on the WS ``nav_step`` event via ``containersLive``.
 * graph_state supports add/remove/edit; other containers are read-only.
 */

import { useState, useCallback, useMemo, useEffect } from "react";
import { Database, Plus, X, Trash2 } from "lucide-react";
import clsx from "clsx";
import { useFlowStore } from "../useFlowStore";
import { api } from "../../api";
import type { ContainerDef, StateDef } from "../types";

const STATE_TYPE_LABELS: Record<string, string> = {
  accumulator: "Accumulator",
  lastWrite: "Last Write",
  counter: "Counter",
  ephemeral: "Ephemeral",
};

const STATE_TYPE_COLORS: Record<string, string> = {
  accumulator: "text-amber-400",
  lastWrite: "text-blue-400",
  counter: "text-green-400",
  ephemeral: "text-gray-400",
};

const STATE_TYPE_BG: Record<string, string> = {
  accumulator: "bg-amber-500/10",
  lastWrite: "bg-blue-500/10",
  counter: "bg-green-500/10",
  ephemeral: "bg-gray-500/10",
};

// Lifetime declares when a state clears — orthogonal to the reducer.
// Maps 1:1 to the backend ``LIFETIME_TO_SIGNALS`` table in
// ``app/agent_loop/state_containers.py``.
const LIFETIME_OPTIONS: { value: string; label: string; hint: string }[] = [
  { value: "forever", label: "Forever", hint: "never clears" },
  { value: "step", label: "Step", hint: "clears at IterOut" },
  { value: "episode", label: "Episode", hint: "clears on episode change" },
  { value: "run", label: "Run", hint: "clears at run end" },
  { value: "custom", label: "Custom", hint: "explicit signal list" },
];

const LIFETIME_COLORS: Record<string, string> = {
  forever: "text-violet-400 bg-violet-500/10",
  step: "text-gray-400 bg-gray-500/10",
  episode: "text-emerald-400 bg-emerald-500/10",
  run: "text-cyan-400 bg-cyan-500/10",
  custom: "text-pink-400 bg-pink-500/10",
};

// One display row. ``live`` is the per-state preview from the backend
// (``get_preview()``): may carry size / value / preview fields.
interface StateRow {
  name: string;
  type: string;
  valueType: string;
  lifetime?: string;
  live?: Record<string, unknown>;
}

function previewToValue(live?: Record<string, unknown>): string {
  if (!live) return "—";
  if (live.size !== undefined) return `${live.size} items`;
  if (live.value !== undefined) return `${live.value}`;
  if (live.preview) return String(live.preview).slice(0, 60);
  return "—";
}

/** A single container's states table, with an Owner column. */
function StatesTable({
  rows,
  owner,
  onRemove,
}: {
  rows: StateRow[];
  owner: string;
  onRemove?: (name: string) => void;
}) {
  if (rows.length === 0) {
    return (
      <div className="p-4 text-center text-xs italic text-gray-600">
        No states defined.
      </div>
    );
  }
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-gray-800 text-left text-[10px] uppercase tracking-wider text-gray-600">
          <th className="px-3 py-1">Owner</th>
          <th className="px-3 py-1">Name</th>
          <th className="px-3 py-1">Type</th>
          <th className="px-3 py-1">Lifetime</th>
          <th className="px-3 py-1">Value Type</th>
          <th className="px-3 py-1">Value</th>
          <th className="w-8 px-1 py-1" />
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.name}
            className="group border-b border-gray-800/50 hover:bg-gray-800/30"
          >
            <td className="px-3 py-1.5">
              <span
                className={clsx(
                  "rounded px-1.5 py-0.5 text-[10px]",
                  owner === "home"
                    ? "bg-amber-500/10 text-amber-300"
                    : "bg-indigo-500/10 text-indigo-300",
                )}
                title={
                  owner === "home"
                    ? "graph / executor process"
                    : "nodeset-owned (subprocess)"
                }
              >
                {owner}
              </span>
            </td>
            <td className="px-3 py-1.5 font-medium text-gray-300">{r.name}</td>
            <td className="px-3 py-1.5">
              <span
                className={clsx(
                  "rounded px-1.5 py-0.5 text-[10px]",
                  STATE_TYPE_BG[r.type],
                  STATE_TYPE_COLORS[r.type],
                )}
              >
                {STATE_TYPE_LABELS[r.type] || r.type}
              </span>
            </td>
            <td className="px-3 py-1.5">
              {r.lifetime ? (
                <span
                  className={clsx(
                    "rounded px-1.5 py-0.5 text-[10px]",
                    LIFETIME_COLORS[r.lifetime] || LIFETIME_COLORS.forever,
                  )}
                  title={
                    LIFETIME_OPTIONS.find((o) => o.value === r.lifetime)?.hint
                  }
                >
                  {r.lifetime}
                </span>
              ) : (
                <span className="text-gray-600">—</span>
              )}
            </td>
            <td className="px-3 py-1.5 text-gray-500">{r.valueType}</td>
            <td className="px-3 py-1.5 font-mono text-gray-500">
              {previewToValue(r.live)}
            </td>
            <td className="px-1 py-1.5">
              {onRemove && (
                <button
                  onClick={() => onRemove(r.name)}
                  className="hidden rounded p-0.5 text-gray-600 hover:text-red-400 group-hover:block"
                >
                  <X size={10} />
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function StatePanel() {
  const canvasStack = useFlowStore((s) => s.canvasStack);
  const nodes = useFlowStore((s) => s.nodes);
  const rootGraphState = useFlowStore((s) => s.graphState);
  const preview = useFlowStore((s) => s.graphStatePreview);
  const containersLive = useFlowStore((s) => s.containersLive);
  const setGraphState = useFlowStore((s) => s.setGraphState);

  // Stack-aware: inside a composite, show its subgraph's "graph_state"
  // container (the well-known id used after ADR-026).
  const graphState = useMemo(() => {
    if (canvasStack.length === 0) return rootGraphState;
    const compositeId = canvasStack[canvasStack.length - 1].nodeId;
    const compositeNode = nodes.find((n) => n.id === compositeId);
    const subgraph = compositeNode?.data?.subgraph as
      | import("../types").GraphDefinition
      | undefined;
    const gs = (subgraph?.containers || []).find((c) => c.id === "graph_state");
    return (gs as ContainerDef | undefined) ?? null;
  }, [canvasStack, nodes, rootGraphState]);

  // Live values for graph_state: prefer the unified WS feed, fall back to the
  // legacy graphStatePreview slice.
  const gsLive =
    containersLive?.["graph_state"]?.states ?? preview ?? undefined;

  // Declared nodeset-owned containers (static schema from loaded nodesets) —
  // shown even before/without a run; live values overlay during a run. Fetched
  // from GET /api/components/nodesets (carries each loaded nodeset's manifest
  // containers). Refetched on tab change + once shortly after, to catch the
  // async nodeset load that follows opening a graph.
  const activeTabId = useFlowStore((s) => s.activeTabId);
  const [declaredNs, setDeclaredNs] = useState<
    Array<{
      id: string;
      label: string;
      owner: string;
      states: Record<string, Record<string, unknown>>;
    }>
  >([]);
  useEffect(() => {
    let cancelled = false;
    const fetchDeclared = () => {
      api
        .listNodesets()
        .then((list) => {
          if (cancelled) return;
          const decl: Array<{
            id: string;
            label: string;
            owner: string;
            states: Record<string, Record<string, unknown>>;
          }> = [];
          for (const ns of list) {
            if (!ns.loaded || !ns.containers) continue;
            for (const c of ns.containers) {
              if (!c.id || c.id === "graph_state") continue;
              decl.push({
                id: c.id,
                label: c.label || c.id,
                owner: ns.name,
                states: (c.states || {}) as Record<
                  string,
                  Record<string, unknown>
                >,
              });
            }
          }
          setDeclaredNs(decl);
        })
        .catch(() => {});
    };
    fetchDeclared();
    const t = setTimeout(fetchDeclared, 2000); // catch async nodeset load
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [activeTabId]);

  // Every container besides the editable graph_state: declared nodeset-owned
  // schemas first (static), with live per-key values overlaid during a run.
  const otherContainers = useMemo(() => {
    const byId: Record<
      string,
      { label: string; owner: string; states: Record<string, unknown> }
    > = {};
    for (const dc of declaredNs) {
      byId[dc.id] = {
        label: dc.label,
        owner: dc.owner,
        states: { ...dc.states },
      };
    }
    for (const [cid, c] of Object.entries(containersLive || {})) {
      if (cid === "graph_state") continue;
      const live = c as {
        label?: string;
        owner?: string;
        states?: Record<string, unknown>;
      };
      const base = byId[cid] || {
        label: live.label || cid,
        owner: live.owner || "nodeset",
        states: {},
      };
      base.label = live.label || base.label;
      base.owner = live.owner || base.owner;
      base.states = { ...base.states, ...(live.states || {}) };
      byId[cid] = base;
    }
    return Object.entries(byId);
  }, [declaredNs, containersLive]);

  const [addingState, setAddingState] = useState(false);
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState("accumulator");
  const [newValueType, setNewValueType] = useState("TEXT");
  const [newLifetime, setNewLifetime] =
    useState<StateDef["lifetime"]>("forever");

  const handleCreate = useCallback(() => {
    setGraphState({
      id: `gs_${Date.now().toString(36)}`,
      label: "Graph State",
      position: { x: 0, y: 0 },
      states: {},
    });
  }, [setGraphState]);

  const handleAddState = useCallback(() => {
    if (!graphState || !newName.trim()) return;
    const entry: StateDef = {
      type: newType as StateDef["type"],
      value_type: newValueType,
      lifetime: newLifetime,
    };
    setGraphState({
      ...graphState,
      states: { ...graphState.states, [newName.trim()]: entry },
    });
    setNewName("");
    setNewLifetime("forever");
    setAddingState(false);
  }, [graphState, newName, newType, newValueType, newLifetime, setGraphState]);

  const handleRemoveState = useCallback(
    (name: string) => {
      if (!graphState) return;
      const { [name]: _, ...rest } = graphState.states;
      setGraphState({ ...graphState, states: rest });
    },
    [graphState, setGraphState],
  );

  // graph_state rows (editable, owner = home).
  const gsRows: StateRow[] = useMemo(
    () =>
      Object.entries(graphState?.states || {}).map(([name, entry]) => ({
        name,
        type: entry.type,
        valueType: entry.value_type,
        lifetime: entry.lifetime || "forever",
        live: gsLive?.[name] as Record<string, unknown> | undefined,
      })),
    [graphState, gsLive],
  );

  // Nothing to show and no graph_state — offer to create one.
  if (!graphState && otherContainers.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <button
          onClick={handleCreate}
          className="flex items-center gap-2 rounded-lg border border-dashed border-gray-700 px-4 py-3 text-sm text-gray-500 transition hover:border-amber-500/50 hover:text-amber-400"
        >
          <Database size={16} />
          Add Graph State
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-auto">
      {/* ── graph_state (editable, owner = home) ── */}
      {graphState && (
        <div className="flex flex-col">
          <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-1.5">
            <Database size={12} className="text-amber-400" />
            <span className="text-xs font-medium text-amber-300">
              {graphState.label || "Graph State"}
            </span>
            <span className="text-[10px] text-gray-600">
              {gsRows.length} states · home
            </span>
            <div className="flex-1" />
            <button
              onClick={() => setAddingState(true)}
              className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-gray-500 transition hover:bg-gray-800 hover:text-amber-400"
            >
              <Plus size={10} /> Add
            </button>
            <button
              onClick={() => setGraphState(null)}
              className="rounded px-1.5 py-0.5 text-[10px] text-gray-600 transition hover:bg-gray-800 hover:text-red-400"
              title="Remove graph state"
            >
              <Trash2 size={10} />
            </button>
          </div>

          {addingState && (
            <div className="flex items-center gap-1.5 border-b border-gray-800 bg-gray-800/30 px-3 py-1.5">
              <input
                type="text"
                placeholder="name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="w-28 rounded border border-gray-700 bg-gray-900 px-1.5 py-0.5 text-xs text-gray-200 focus:border-amber-500 focus:outline-none"
                autoFocus
                onKeyDown={(e) => e.key === "Enter" && handleAddState()}
              />
              <select
                value={newType}
                onChange={(e) => setNewType(e.target.value)}
                className="rounded border border-gray-700 bg-gray-900 px-1 py-0.5 text-xs text-gray-300"
                title="Reducer — how writes combine"
              >
                <option value="accumulator">Accumulator</option>
                <option value="lastWrite">Last Write</option>
                <option value="counter">Counter</option>
                <option value="ephemeral">Ephemeral</option>
              </select>
              <select
                value={newLifetime}
                onChange={(e) =>
                  setNewLifetime(e.target.value as StateDef["lifetime"])
                }
                className="rounded border border-gray-700 bg-gray-900 px-1 py-0.5 text-xs text-gray-300"
                title="Lifetime — when the state clears"
              >
                {LIFETIME_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value} title={opt.hint}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <input
                type="text"
                placeholder="value type"
                value={newValueType}
                onChange={(e) => setNewValueType(e.target.value)}
                className="w-20 rounded border border-gray-700 bg-gray-900 px-1.5 py-0.5 text-xs text-gray-300"
                onKeyDown={(e) => e.key === "Enter" && handleAddState()}
              />
              <button
                onClick={handleAddState}
                disabled={!newName.trim()}
                className="rounded bg-amber-600 px-2 py-0.5 text-xs text-white hover:bg-amber-500 disabled:bg-gray-700 disabled:text-gray-500"
              >
                Add
              </button>
              <button
                onClick={() => setAddingState(false)}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                Cancel
              </button>
            </div>
          )}

          <StatesTable
            rows={gsRows}
            owner="home"
            onRemove={handleRemoveState}
          />
        </div>
      )}

      {/* ── other live containers (read-only): home + nodeset-owned ── */}
      {otherContainers.map(([cid, c]) => {
        const rows: StateRow[] = Object.entries(c.states || {}).map(
          ([name, live]) => {
            const l = live as Record<string, unknown>;
            return {
              name,
              type: String(l.type ?? "lastWrite"),
              valueType: String(l.value_type ?? "ANY"),
              lifetime: l.lifetime ? String(l.lifetime) : undefined,
              live: l,
            };
          },
        );
        return (
          <div key={cid} className="flex flex-col">
            <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-1.5">
              <Database size={12} className="text-indigo-400" />
              <span className="text-xs font-medium text-indigo-300">
                {c.label || cid}
              </span>
              <span className="text-[10px] text-gray-600">
                {rows.length} states · {c.owner}
              </span>
            </div>
            <StatesTable rows={rows} owner={c.owner} />
          </div>
        );
      })}
    </div>
  );
}
