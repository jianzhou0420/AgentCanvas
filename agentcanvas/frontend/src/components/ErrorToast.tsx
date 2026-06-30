/** Floating toast layer — auto-pops on new error/warning envelopes.
 *
 * Mounted once at the app root. Listens to `useErrorStore.entries` and
 * surfaces fresh items (info/debug never toast — they only live in the
 * Report tab). Click the toast body → opens the Report tab.
 */

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { X, AlertCircle, AlertTriangle } from "lucide-react";
import { useErrorStore, type ErrorEntry } from "../errorStore";

const MAX_VISIBLE = 3;
const DISMISS_MS: Record<"error" | "warning", number> = {
  error: 6000,
  warning: 4000,
};

interface Toast {
  entry: ErrorEntry;
  /** seq from store at the moment we surfaced it (for keying). */
  seq: number;
}

export default function ErrorToast() {
  const entries = useErrorStore((s) => s.entries);
  const focusReport = useErrorStore((s) => s.focusReport);

  const [toasts, setToasts] = useState<Toast[]>([]);
  // Track highest seq we've already shown so we never re-pop on store edits
  // (mark-read, dismiss, pin etc. all mutate `entries`).
  const lastSeenSeq = useRef(0);

  useEffect(() => {
    if (entries.length === 0) return;
    const fresh: Toast[] = [];
    for (const e of entries) {
      if (e.seq <= lastSeenSeq.current) break; // entries are newest-first
      if (e.severity !== "error" && e.severity !== "warning") continue;
      fresh.push({ entry: e, seq: e.seq });
    }
    if (fresh.length === 0) {
      // Still bump tracker if newer non-toastable entries arrived.
      if (entries[0] && entries[0].seq > lastSeenSeq.current) {
        lastSeenSeq.current = entries[0].seq;
      }
      return;
    }
    lastSeenSeq.current = Math.max(lastSeenSeq.current, fresh[0].seq);
    // Reverse so older fresh entries appear above newer in the stack.
    setToasts((cur) => [...fresh.reverse(), ...cur].slice(0, MAX_VISIBLE));
  }, [entries]);

  // Auto-dismiss timers
  useEffect(() => {
    if (toasts.length === 0) return;
    const timers = toasts.map((t) => {
      const ms = DISMISS_MS[t.entry.severity as "error" | "warning"] ?? 5000;
      return window.setTimeout(() => {
        setToasts((cur) => cur.filter((x) => x.seq !== t.seq));
      }, ms);
    });
    return () => timers.forEach((id) => window.clearTimeout(id));
  }, [toasts]);

  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-[100] flex w-[360px] flex-col gap-2">
      {toasts.map((t) => {
        const isError = t.entry.severity === "error";
        const Icon = isError ? AlertCircle : AlertTriangle;
        return (
          <div
            key={t.seq}
            onClick={() => {
              focusReport();
              setToasts((cur) => cur.filter((x) => x.seq !== t.seq));
            }}
            className={clsx(
              "pointer-events-auto cursor-pointer rounded-md border shadow-lg backdrop-blur-sm transition",
              isError
                ? "border-red-500/40 bg-red-950/90 hover:bg-red-900/90"
                : "border-amber-500/40 bg-amber-950/90 hover:bg-amber-900/90",
            )}
          >
            <div className="flex items-start gap-2 p-3">
              <Icon
                size={14}
                className={clsx(
                  "mt-0.5 shrink-0",
                  isError ? "text-red-400" : "text-amber-400",
                )}
              />
              <div className="min-w-0 flex-1">
                <div
                  className={clsx(
                    "truncate text-xs font-semibold",
                    isError ? "text-red-200" : "text-amber-200",
                  )}
                  title={t.entry.title}
                >
                  {t.entry.title}
                </div>
                {t.entry.message && t.entry.message !== t.entry.title && (
                  <div className="mt-0.5 line-clamp-2 text-[11px] text-gray-300">
                    {t.entry.message}
                  </div>
                )}
                <div className="mt-1 font-mono text-[9px] uppercase tracking-wider text-gray-500">
                  {t.entry.source} · {t.entry.code}
                </div>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setToasts((cur) => cur.filter((x) => x.seq !== t.seq));
                }}
                className="shrink-0 rounded p-0.5 text-gray-500 hover:bg-gray-800 hover:text-gray-200"
                title="Dismiss"
              >
                <X size={12} />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
