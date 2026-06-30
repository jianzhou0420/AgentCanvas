/** Execution log panel — shows per-node inputs/outputs/timing per step.
 *
 * Two view modes:
 * - Timeline: step-by-step, all nodes that fired per step, expandable
 * - Node: click a node → see its history across all steps
 *
 * Subscribes to `exec_log` WS events for real-time updates.
 * Falls back to REST `GET /api/logs/{id}` for historical runs.
 */

import { useEffect, useState, useRef } from "react";
import clsx from "clsx";
import { wsManager } from "../../ws";
import { useStore } from "../../store";

// ── Types ──

interface LogEntry {
  timestamp: string;
  execution_id: string;
  source: string;
  step: number;
  node_id: string;
  node_type: string;
  node_label: string;
  duration_ms: number;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  inner_log: { key: string; value: unknown }[];
  error: string | null;
}

// ── Helpers ──

function formatDuration(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function truncateValue(val: unknown, maxLen = 120): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "object") {
    const obj = val as Record<string, unknown>;
    if (obj.__type === "large_string") {
      return `[string ${obj.length} chars] ${obj.preview || ""}…`;
    }
    // LIST[T] payload (ADR-027) — show "[N INNER(s)]"
    if (obj.__type === "list") {
      const wt = String(obj.wire_type ?? "LIST[?]");
      const inner =
        wt.startsWith("LIST[") && wt.endsWith("]") ? wt.slice(5, -1) : "ITEM";
      const count = obj.count ?? 0;
      return `[${count} ${inner}${count === 1 ? "" : "s"}]`;
    }
    if (obj.__type) {
      return `<${obj.__type}${obj.shape ? " " + JSON.stringify(obj.shape) : ""}>`;
    }
    const s = JSON.stringify(val);
    return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
  }
  const s = String(val);
  return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
}

// ── Expandable Value ──

function ExpandableValue({ label, value }: { label: string; value: unknown }) {
  const [expanded, setExpanded] = useState(false);
  const preview = truncateValue(value);
  const full =
    typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  const isLong = full.length > 120;

  return (
    <div className="mb-1">
      <span className="text-gray-500 text-[10px]">{label}: </span>
      {isLong ? (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-blue-400 hover:text-blue-300 text-[10px] underline"
          >
            {expanded ? "collapse" : "expand"}
          </button>
          {expanded ? (
            <pre className="mt-0.5 text-[10px] text-gray-300 bg-gray-800 p-1 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
              {full}
            </pre>
          ) : (
            <span className="text-gray-400 text-[10px] ml-1">{preview}</span>
          )}
        </>
      ) : (
        <span className="text-gray-400 text-[10px] ml-1">{preview}</span>
      )}
    </div>
  );
}

// ── Entry Card ──

function LogEntryCard({ entry }: { entry: LogEntry }) {
  const [showDetails, setShowDetails] = useState(false);
  const hasInner = entry.inner_log && entry.inner_log.length > 0;

  return (
    <div
      className={clsx(
        "border-l-2 pl-2 py-1 mb-1 cursor-pointer hover:bg-gray-800/50 transition",
        entry.error ? "border-red-500" : "border-gray-700",
      )}
      onClick={() => setShowDetails(!showDetails)}
    >
      {/* Header line */}
      <div className="flex items-center gap-2 text-[11px]">
        <span className="text-gray-500 w-8 text-right shrink-0">
          #{entry.step}
        </span>
        <span
          className={clsx(
            "font-medium truncate",
            entry.error ? "text-red-400" : "text-gray-200",
          )}
        >
          {entry.node_label}
        </span>
        <span className="text-gray-600 text-[10px] truncate">
          {entry.node_type}
        </span>
        <span className="ml-auto text-gray-500 text-[10px] shrink-0">
          {formatDuration(entry.duration_ms)}
        </span>
      </div>

      {/* Error */}
      {entry.error && (
        <div className="text-red-400 text-[10px] mt-0.5 truncate">
          {entry.error}
        </div>
      )}

      {/* Expanded details */}
      {showDetails && (
        <div className="mt-1 ml-10 space-y-1">
          {/* Inputs */}
          {Object.keys(entry.inputs).length > 0 && (
            <div>
              <div className="text-[10px] text-gray-500 font-semibold mb-0.5">
                INPUTS
              </div>
              {Object.entries(entry.inputs).map(([k, v]) => (
                <ExpandableValue key={k} label={k} value={v} />
              ))}
            </div>
          )}

          {/* Outputs */}
          {Object.keys(entry.outputs).length > 0 && (
            <div>
              <div className="text-[10px] text-gray-500 font-semibold mb-0.5">
                OUTPUTS
              </div>
              {Object.entries(entry.outputs).map(([k, v]) => (
                <ExpandableValue key={k} label={k} value={v} />
              ))}
            </div>
          )}

          {/* Inner log (voluntary _self_log entries) */}
          {hasInner && (
            <div>
              <div className="text-[10px] text-amber-500 font-semibold mb-0.5">
                NODE LOG
              </div>
              {entry.inner_log.map((item, i) => (
                <ExpandableValue key={i} label={item.key} value={item.value} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Panel ──

export default function LogPanel() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScroll = useRef(true);

  // Subscribe to exec_log WS events
  useEffect(() => {
    const unsub = wsManager.on("exec_log", (data) => {
      const entry = data as LogEntry;
      setEntries((prev) => {
        const next = [...prev, entry];
        // Keep last 500 entries in UI
        return next.length > 500 ? next.slice(-500) : next;
      });
    });
    return unsub;
  }, []);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (autoScroll.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  // Handle scroll — disable auto-scroll if user scrolled up
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    autoScroll.current = scrollHeight - scrollTop - clientHeight < 40;
  };

  // Filter entries
  const filtered = filter
    ? entries.filter(
        (e) =>
          e.node_label.toLowerCase().includes(filter.toLowerCase()) ||
          e.node_type.toLowerCase().includes(filter.toLowerCase()),
      )
    : entries;

  // Group by step
  const steps = new Map<number, LogEntry[]>();
  for (const e of filtered) {
    const list = steps.get(e.step) || [];
    list.push(e);
    steps.set(e.step, list);
  }

  return (
    <div className="flex h-full flex-col text-sm">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-2 py-1 border-b border-gray-800">
        <input
          type="text"
          placeholder="Filter nodes…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex-1 bg-gray-800 text-gray-300 text-xs px-2 py-0.5 rounded border border-gray-700 focus:border-blue-500 outline-none"
        />
        <span className="text-gray-600 text-[10px]">
          {filtered.length} entries
        </span>
        <button
          onClick={() => setEntries([])}
          className="text-gray-500 hover:text-gray-300 text-[10px] px-1"
          title="Clear log"
        >
          Clear
        </button>
        <button
          onClick={() => useStore.getState().setAppMode("logs")}
          className="text-blue-400 hover:text-blue-300 text-[10px] px-1"
          title="Open in Log Viewer"
        >
          Open
        </button>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-2 py-1"
      >
        {filtered.length === 0 ? (
          <div className="text-gray-600 text-xs text-center mt-8">
            No log entries yet. Run a graph to see node execution logs.
          </div>
        ) : (
          Array.from(steps.entries()).map(([step, stepEntries]) => (
            <div key={step} className="mb-2">
              <div className="text-[10px] text-gray-600 font-semibold mb-0.5 sticky top-0 bg-gray-900 py-0.5">
                Step {step}
                <span className="ml-2 text-gray-700">
                  ({stepEntries.length} node{stepEntries.length > 1 ? "s" : ""})
                </span>
              </div>
              {stepEntries.map((entry, i) => (
                <LogEntryCard
                  key={`${entry.node_id}-${entry.step}-${i}`}
                  entry={entry}
                />
              ))}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
