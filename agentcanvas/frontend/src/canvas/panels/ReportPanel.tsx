/** Bottom-panel "Report" tab — global event console.
 *
 * Renders the time-ordered list of envelopes from `useErrorStore`. Each
 * row collapses to one line (severity dot, source pill, title, time);
 * click to expand details / traceback / scope. Top toolbar carries
 * severity filter chips + Mark-all-read + Clear.
 */

import { useMemo, useState } from "react";
import clsx from "clsx";
import {
  Trash2,
  CheckCheck,
  Pin,
  PinOff,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useErrorStore, type ErrorEntry } from "../../errorStore";
import type { ErrorSeverity } from "../../errors";

const SEVERITY_ORDER: ErrorSeverity[] = ["error", "warning", "info", "debug"];

const SEVERITY_STYLES: Record<
  ErrorSeverity,
  { dot: string; text: string; bg: string }
> = {
  error: { dot: "bg-red-500", text: "text-red-300", bg: "bg-red-500/10" },
  warning: {
    dot: "bg-amber-500",
    text: "text-amber-300",
    bg: "bg-amber-500/10",
  },
  info: { dot: "bg-sky-500", text: "text-sky-300", bg: "bg-sky-500/10" },
  debug: { dot: "bg-gray-500", text: "text-gray-400", bg: "bg-gray-500/10" },
};

function fmtTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return ts;
  }
}

function ReportRow({ entry }: { entry: ErrorEntry }) {
  const [expanded, setExpanded] = useState(false);
  const sev = SEVERITY_STYLES[entry.severity];
  const markRead = useErrorStore((s) => s.markRead);
  const dismiss = useErrorStore((s) => s.dismiss);
  const pin = useErrorStore((s) => s.pin);

  const toggle = () => {
    setExpanded((v) => !v);
    if (!entry.read) markRead(entry.id);
  };

  const traceback =
    typeof entry.details?.traceback === "string"
      ? (entry.details.traceback as string)
      : null;

  return (
    <div
      className={clsx(
        "border-b border-gray-800/60 transition",
        !entry.read && "bg-gray-800/40",
      )}
    >
      <div
        onClick={toggle}
        className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-xs hover:bg-gray-800/60"
      >
        {expanded ? (
          <ChevronDown size={11} className="shrink-0 text-gray-500" />
        ) : (
          <ChevronRight size={11} className="shrink-0 text-gray-500" />
        )}
        <span className={clsx("h-2 w-2 shrink-0 rounded-full", sev.dot)} />
        <span className="w-16 shrink-0 font-mono text-[10px] text-gray-500">
          {fmtTime(entry.ts)}
        </span>
        <span
          className={clsx(
            "shrink-0 rounded px-1.5 py-0.5 font-mono text-[9px] uppercase",
            sev.bg,
            sev.text,
          )}
        >
          {entry.source}
        </span>
        <span className="truncate text-gray-300" title={entry.title}>
          {entry.title}
        </span>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              pin(entry.id, !entry.pinned);
            }}
            className="rounded p-0.5 text-gray-500 hover:bg-gray-700 hover:text-gray-300"
            title={entry.pinned ? "Unpin" : "Pin"}
          >
            {entry.pinned ? <Pin size={11} /> : <PinOff size={11} />}
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              dismiss(entry.id);
            }}
            className="rounded p-0.5 text-gray-500 hover:bg-gray-700 hover:text-red-400"
            title="Dismiss"
          >
            <Trash2 size={11} />
          </button>
        </div>
      </div>
      {expanded && (
        <div className="border-l-2 border-gray-700 bg-gray-950/60 px-4 py-2 text-[11px]">
          <div className="mb-1 font-mono text-gray-400">
            <span className="text-gray-500">code:</span> {entry.code}
          </div>
          {entry.message && entry.message !== entry.title && (
            <div className="mb-2 whitespace-pre-wrap text-gray-300">
              {entry.message}
            </div>
          )}
          {Object.keys(entry.scope).length > 0 && (
            <div className="mb-2">
              <div className="mb-0.5 text-[10px] uppercase tracking-wider text-gray-500">
                scope
              </div>
              <pre className="overflow-x-auto rounded bg-gray-900 p-2 text-[10px] text-gray-400">
                {JSON.stringify(entry.scope, null, 2)}
              </pre>
            </div>
          )}
          {traceback && (
            <div className="mb-2">
              <div className="mb-0.5 text-[10px] uppercase tracking-wider text-gray-500">
                traceback
              </div>
              <pre className="max-h-64 overflow-auto rounded bg-gray-900 p-2 text-[10px] text-gray-400">
                {traceback}
              </pre>
            </div>
          )}
          {entry.hint && (
            <div className="mt-1 rounded bg-blue-500/10 px-2 py-1 text-[11px] text-blue-300">
              hint: {entry.hint}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ReportPanel() {
  const entries = useErrorStore((s) => s.entries);
  const markAllRead = useErrorStore((s) => s.markAllRead);
  const clear = useErrorStore((s) => s.clear);

  const [filter, setFilter] = useState<Set<ErrorSeverity>>(
    () => new Set(SEVERITY_ORDER),
  );

  const visible = useMemo(
    () => entries.filter((e) => filter.has(e.severity)),
    [entries, filter],
  );

  const toggleSeverity = (s: ErrorSeverity) => {
    setFilter((cur) => {
      const next = new Set(cur);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  const counts = useMemo(() => {
    const c: Record<ErrorSeverity, number> = {
      error: 0,
      warning: 0,
      info: 0,
      debug: 0,
    };
    for (const e of entries) c[e.severity]++;
    return c;
  }, [entries]);

  return (
    <div className="flex h-full flex-col bg-gray-900">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-gray-800 bg-gray-900 px-3 py-1.5">
        {SEVERITY_ORDER.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => toggleSeverity(s)}
            className={clsx(
              "flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wider transition",
              filter.has(s)
                ? clsx(SEVERITY_STYLES[s].bg, SEVERITY_STYLES[s].text)
                : "bg-gray-800 text-gray-600 hover:text-gray-400",
            )}
            title={`Toggle ${s}`}
          >
            <span
              className={clsx(
                "h-1.5 w-1.5 rounded-full",
                SEVERITY_STYLES[s].dot,
              )}
            />
            {s} {counts[s] > 0 && <span>({counts[s]})</span>}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={markAllRead}
            className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-gray-400 hover:bg-gray-800 hover:text-gray-200"
            title="Mark all read"
          >
            <CheckCheck size={11} /> read
          </button>
          <button
            type="button"
            onClick={clear}
            className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-gray-400 hover:bg-gray-800 hover:text-red-400"
            title="Clear all (keeps pinned)"
          >
            <Trash2 size={11} /> clear
          </button>
        </div>
      </div>
      {/* List */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-gray-600">
            {entries.length === 0
              ? "No events yet — errors, warnings and logs will appear here."
              : "No events match the active filters."}
          </div>
        ) : (
          visible.map((e) => <ReportRow key={e.id} entry={e} />)
        )}
      </div>
    </div>
  );
}
