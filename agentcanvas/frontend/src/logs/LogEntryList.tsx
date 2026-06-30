/** Virtualized log entry list with expandable detail.
 * Uses @tanstack/react-virtual for smooth scrolling of 2000+ entries.
 * Selecting an entry shows full data in a fixed-width detail pane (master-detail layout).
 */

import { useRef, useState, useEffect, useCallback, Suspense } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import clsx from "clsx";
import { logApi } from "./logApi";
import { resolveRenderer, ErrorRenderer } from "./renderers/registry";
import type { LogEntry } from "./types";
import type { LogFilterState } from "./LogFilters";
import { LogContext } from "./LogContext";
import type { LogViewMode } from "./LogContext";
import { getNodeColor } from "./nodeColors";

interface Props {
  executionId: string;
  filters: LogFilterState;
  viewMode: LogViewMode;
  onFilteredCount: (count: number) => void;
  onNodeTypes: (types: string[]) => void;
}

function formatDuration(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Render a single entry's inputs or outputs using the type registry. */
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

/** Compact inline preview for entry rows — shows data summaries between node label and duration. */
function InlinePreview({
  outputs,
  portWireTypes,
  executionId,
}: {
  outputs: Record<string, unknown>;
  portWireTypes: Record<string, string>;
  executionId: string;
}) {
  const previews: React.ReactNode[] = [];

  for (const [key, val] of Object.entries(outputs)) {
    if (val === null || val === undefined) continue;
    const wt = portWireTypes[key];
    const obj =
      typeof val === "object" && !Array.isArray(val)
        ? (val as Record<string, unknown>)
        : null;

    // Asset image thumbnail (24x24)
    if (obj?.__type === "asset" && typeof obj.path === "string") {
      previews.push(
        <img
          key={key}
          src={logApi.assetUrl(executionId, String(obj.path))}
          alt={key}
          className="h-6 w-6 rounded object-cover border border-gray-700"
          loading="lazy"
        />,
      );
      continue;
    }

    // Asset group (OBSERVATION) — show rgb thumb
    if (
      obj?.__type === "asset_group" &&
      obj.rgb &&
      typeof obj.rgb === "object"
    ) {
      const rgb = obj.rgb as Record<string, unknown>;
      if (rgb.__type === "asset" && typeof rgb.path === "string") {
        previews.push(
          <img
            key={key}
            src={logApi.assetUrl(executionId, String(rgb.path))}
            alt={key}
            className="h-6 w-6 rounded object-cover border border-gray-700"
            loading="lazy"
          />,
        );
        continue;
      }
    }

    // ACTION badge (compact)
    if (wt === "ACTION" && typeof val === "number") {
      const labels: Record<number, { text: string; cls: string }> = {
        0: { text: "STOP", cls: "bg-red-600/30 text-red-300" },
        1: { text: "FWD", cls: "bg-green-600/30 text-green-300" },
        2: { text: "LEFT", cls: "bg-blue-600/30 text-blue-300" },
        3: { text: "RIGHT", cls: "bg-orange-600/30 text-orange-300" },
      };
      const a = labels[val];
      if (a) {
        previews.push(
          <span
            key={key}
            className={clsx(
              "rounded px-1 py-0.5 text-[9px] font-medium",
              a.cls,
            )}
          >
            {a.text}
          </span>,
        );
        continue;
      }
    }

    // TEXT snippet
    if (wt === "TEXT" && typeof val === "string" && val.length > 0) {
      previews.push(
        <span key={key} className="text-gray-500 text-[10px] truncate max-w-32">
          {val.slice(0, 60)}
        </span>,
      );
      continue;
    }

    // Large string preview
    if (obj?.__type === "large_string" && obj.preview) {
      previews.push(
        <span key={key} className="text-gray-500 text-[10px] truncate max-w-32">
          {String(obj.preview).slice(0, 60)}
        </span>,
      );
      continue;
    }

    // METRICS — top 1-2 values
    if (wt === "METRICS" && obj && !obj.__type) {
      const nums = Object.entries(obj)
        .filter(([, v]) => typeof v === "number")
        .slice(0, 2);
      if (nums.length > 0) {
        previews.push(
          <span key={key} className="text-[9px] text-gray-500">
            {nums.map(([k, v]) => `${k}=${(v as number).toFixed(2)}`).join(" ")}
          </span>,
        );
        continue;
      }
    }

    // Limit to 3 previews per row to avoid clutter
    if (previews.length >= 3) break;
  }

  if (previews.length === 0) return null;
  return <div className="flex items-center gap-1 shrink-0">{previews}</div>;
}

/** Inline expanded detail for a single entry — used in detail mode. */
function DetailInline({ entry }: { entry: LogEntry }) {
  const hasInputs = entry.inputs && Object.keys(entry.inputs).length > 0;
  const hasOutputs = entry.outputs && Object.keys(entry.outputs).length > 0;
  const hasInnerLog = entry.inner_log && entry.inner_log.length > 0;

  if (!hasInputs && !hasOutputs && !hasInnerLog && !entry.error) {
    return (
      <div className="px-3 pb-2">
        <span className="text-gray-600 text-[10px]">No data</span>
      </div>
    );
  }

  return (
    <div className="px-3 pb-3 space-y-2">
      {entry.error && (
        <div>
          <div className="text-[10px] text-red-400 font-semibold mb-0.5">
            ERROR
          </div>
          <Suspense
            fallback={
              <span className="text-gray-600 text-[10px]">loading...</span>
            }
          >
            <ErrorRenderer value={entry.error} />
          </Suspense>
        </div>
      )}

      {hasOutputs && (
        <div>
          <div className="text-[10px] text-gray-500 font-semibold mb-1">
            OUTPUTS
          </div>
          <RenderedValues
            data={entry.outputs}
            portWireTypes={entry.port_wire_types ?? {}}
          />
        </div>
      )}

      {hasInputs && (
        <div>
          <div className="text-[10px] text-gray-500 font-semibold mb-1">
            INPUTS
          </div>
          <RenderedValues
            data={entry.inputs}
            portWireTypes={entry.port_wire_types ?? {}}
          />
        </div>
      )}

      {hasInnerLog && (
        <div>
          <div className="text-[10px] text-amber-500 font-semibold mb-1">
            NODE LOG
          </div>
          <div className="space-y-1">
            {entry.inner_log.map((item, i) => {
              const Renderer = resolveRenderer(item.value);
              return (
                <div key={i}>
                  <span className="text-[10px] text-gray-500">
                    {item.key}:{" "}
                  </span>
                  <Suspense
                    fallback={
                      <span className="text-gray-600 text-[10px]">...</span>
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
    </div>
  );
}

export default function LogEntryList({
  executionId,
  filters,
  viewMode,
  onFilteredCount,
  onNodeTypes,
}: Props) {
  const isDetail = viewMode === "detail";
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detailEntry, setDetailEntry] = useState<LogEntry | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const parentRef = useRef<HTMLDivElement>(null);

  // Load entries
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setExpandedId(null);
    setDetailEntry(null);

    logApi
      .getEntries(executionId, { limit: 2000 })
      .then((res) => {
        if (!cancelled) {
          setEntries(res.entries);
          setLoading(false);
          const types = [
            ...new Set(res.entries.map((e) => e.node_type)),
          ].sort();
          onNodeTypes(types);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [executionId, onNodeTypes]);

  // Filter entries
  const filtered = entries.filter((e) => {
    if (filters.errorsOnly && !e.error) return false;
    if (filters.nodeType && e.node_type !== filters.nodeType) return false;
    if (filters.search) {
      const q = filters.search.toLowerCase();
      if (
        !e.node_label.toLowerCase().includes(q) &&
        !e.node_type.toLowerCase().includes(q)
      )
        return false;
    }
    return true;
  });

  useEffect(() => {
    onFilteredCount(filtered.length);
  }, [filtered.length, onFilteredCount]);

  // Virtual scrolling — detail mode rows are taller due to inline expanded content
  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => (isDetail ? 300 : 44),
    overscan: isDetail ? 3 : 10,
  });

  // Expand/collapse with lazy fetch
  const handleExpand = useCallback(
    async (entry: LogEntry) => {
      const entryKey = `${entry.node_id}-${entry.step}`;
      if (expandedId === entryKey) {
        setExpandedId(null);
        setDetailEntry(null);
        return;
      }

      setExpandedId(entryKey);

      // If entry already has inputs/outputs from bulk fetch, use directly
      if (entry.inputs && Object.keys(entry.inputs).length > 0) {
        setDetailEntry(entry);
        return;
      }

      // Lazy fetch full entry
      setDetailLoading(true);
      try {
        const res = await logApi.getEntries(executionId, {
          node_id: entry.node_id,
          step: entry.step,
          limit: 1,
        });
        if (res.entries.length > 0) {
          setDetailEntry(res.entries[0]);
        }
      } catch {
        // ignore
      }
      setDetailLoading(false);
    },
    [expandedId, executionId],
  );

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600 text-xs">
        Loading entries...
      </div>
    );
  }

  if (filtered.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600 text-xs">
        {entries.length === 0
          ? "No log entries. Run a graph to see logs."
          : "No entries match filters."}
      </div>
    );
  }

  return (
    <LogContext.Provider value={{ executionId, viewMode }}>
      <div className="flex flex-1 min-h-0">
        {/* Entry list (virtualized) */}
        <div ref={parentRef} className="flex-1 overflow-y-auto">
          <div
            style={{ height: virtualizer.getTotalSize(), position: "relative" }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const entry = filtered[virtualRow.index];
              const entryKey = `${entry.node_id}-${entry.step}`;
              const isExpanded = expandedId === entryKey;
              const nodeColor = getNodeColor(entry.node_type);

              return (
                <div
                  key={virtualRow.key}
                  data-index={virtualRow.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${virtualRow.start}px)`,
                    padding: isDetail ? "4px 8px 4px 8px" : undefined,
                  }}
                >
                  <div
                    className={clsx(
                      isDetail &&
                        "rounded-md border border-gray-700/60 bg-gray-800/30 overflow-hidden",
                    )}
                  >
                    {/* Row header — always visible */}
                    <button
                      onClick={() => handleExpand(entry)}
                      className={clsx(
                        "w-full flex items-center gap-2 px-3 py-1.5 text-left transition text-[11px]",
                        isExpanded
                          ? "bg-blue-600/10"
                          : clsx("hover:bg-gray-800/30", nodeColor.bg),
                        entry.error && "border-l-2 border-l-red-500",
                        !isDetail && "border-b border-gray-800/50",
                        isDetail &&
                          "border-b border-gray-700/40 bg-gray-800/50",
                      )}
                    >
                      <span
                        className={clsx(
                          "h-1.5 w-1.5 rounded-full shrink-0",
                          nodeColor.dot,
                        )}
                      />
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
                      {!isDetail && (
                        <InlinePreview
                          outputs={entry.outputs || {}}
                          portWireTypes={entry.port_wire_types || {}}
                          executionId={executionId}
                        />
                      )}
                      <span className="ml-auto text-gray-500 text-[10px] shrink-0">
                        {formatDuration(entry.duration_ms)}
                      </span>
                    </button>

                    {/* Inline expanded detail — shown in detail mode for every row */}
                    {isDetail && <DetailInline entry={entry} />}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Side detail pane — only in overall mode when an entry is selected */}
        {!isDetail && expandedId && (
          <div className="w-80 border-l border-gray-800 overflow-y-auto p-3 bg-gray-900/50 shrink-0">
            {detailLoading ? (
              <div className="text-gray-600 text-xs">Loading details...</div>
            ) : detailEntry ? (
              <div className="space-y-3">
                {/* Header */}
                <div>
                  <div className="text-[12px] font-medium text-gray-200">
                    {detailEntry.node_label}
                  </div>
                  <div className="text-[10px] text-gray-500">
                    {detailEntry.node_type} — Step #{detailEntry.step}
                  </div>
                  <div className="text-[10px] text-gray-500">
                    {formatDuration(detailEntry.duration_ms)}
                  </div>
                </div>

                {/* Error */}
                {detailEntry.error && (
                  <div>
                    <div className="text-[10px] text-gray-500 font-semibold mb-0.5">
                      ERROR
                    </div>
                    <Suspense
                      fallback={
                        <span className="text-gray-600 text-[10px]">
                          loading...
                        </span>
                      }
                    >
                      <ErrorRenderer value={detailEntry.error} />
                    </Suspense>
                  </div>
                )}

                {/* Inputs */}
                {Object.keys(detailEntry.inputs).length > 0 && (
                  <div>
                    <div className="text-[10px] text-gray-500 font-semibold mb-1">
                      INPUTS
                    </div>
                    <RenderedValues
                      data={detailEntry.inputs}
                      portWireTypes={detailEntry.port_wire_types ?? {}}
                    />
                  </div>
                )}

                {/* Outputs */}
                {Object.keys(detailEntry.outputs).length > 0 && (
                  <div>
                    <div className="text-[10px] text-gray-500 font-semibold mb-1">
                      OUTPUTS
                    </div>
                    <RenderedValues
                      data={detailEntry.outputs}
                      portWireTypes={detailEntry.port_wire_types ?? {}}
                    />
                  </div>
                )}

                {/* Inner log */}
                {detailEntry.inner_log && detailEntry.inner_log.length > 0 && (
                  <div>
                    <div className="text-[10px] text-amber-500 font-semibold mb-1">
                      NODE LOG
                    </div>
                    <div className="space-y-1">
                      {detailEntry.inner_log.map((item, i) => {
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
              </div>
            ) : (
              <div className="text-gray-600 text-xs">
                Select an entry to view details.
              </div>
            )}
          </div>
        )}
      </div>
    </LogContext.Provider>
  );
}
