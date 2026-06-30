/** Global event/error store — feeds the bottom-panel "Report" tab + toast layer.
 *
 * Producers:
 * - `error_event` WS frames (backend bus)
 * - `api.ts` HTTP error wrapper
 * - React error boundaries
 * - Anywhere a frontend wants to surface a user-visible event
 *
 * Sinks:
 * - `ErrorToast` listens to `entries[0]` for new fresh errors
 * - `ReportPanel` renders the full list
 * - `ExecutionToolbar` reads `unreadCount` for the bell badge
 */

import { create } from "zustand";
import type { ErrorEnvelope, ErrorSeverity, ErrorSource } from "./errors";

const CAP = 200;

export interface ErrorEntry extends ErrorEnvelope {
  read: boolean;
  pinned: boolean;
  /** Monotonic local seq — used so toast can detect "new since last render". */
  seq: number;
}

interface ErrorStore {
  entries: ErrorEntry[]; // newest first
  unreadCount: number;
  nextSeq: number;

  /** Bumped whenever something (e.g. the toolbar bell) wants the Report tab
   *  brought to focus. Bottom-panel components subscribe and react. */
  focusReportTick: number;
  focusReport: () => void;

  ingest: (env: ErrorEnvelope) => void;
  reportLocal: (partial: {
    source?: ErrorSource;
    severity?: ErrorSeverity;
    code?: string;
    title: string;
    message?: string;
    scope?: Record<string, unknown>;
    details?: Record<string, unknown>;
    hint?: string | null;
  }) => void;

  markRead: (id: string) => void;
  markAllRead: () => void;
  pin: (id: string, pinned: boolean) => void;
  dismiss: (id: string) => void;
  clear: () => void;

  /** Backfill from REST after WS reconnect. */
  backfill: (events: ErrorEnvelope[]) => void;
}

function genId(): string {
  return Math.random().toString(36).slice(2, 14);
}

function nowIso(): string {
  return new Date().toISOString();
}

function recompUnread(entries: ErrorEntry[]): number {
  // Badge counts attention-worthy unread only — info/debug fill the Report
  // tab as a developer console but should never inflate the toolbar bell or
  // tab badge. Matches the toast policy (only error/warning auto-pop).
  let n = 0;
  for (const e of entries) {
    if (e.read) continue;
    if (e.severity === "error" || e.severity === "warning") n++;
  }
  return n;
}

export const useErrorStore = create<ErrorStore>((set, get) => ({
  entries: [],
  unreadCount: 0,
  nextSeq: 1,
  focusReportTick: 0,

  focusReport: () => set((s) => ({ focusReportTick: s.focusReportTick + 1 })),

  ingest: (env) => {
    const state = get();
    // Dedup by id — backfill may overlap with live events.
    if (state.entries.some((e) => e.id === env.id)) return;
    const seq = state.nextSeq;
    const entry: ErrorEntry = {
      ...env,
      read: false,
      pinned: false,
      seq,
    };
    const next = [entry, ...state.entries].slice(0, CAP);
    set({
      entries: next,
      unreadCount: recompUnread(next),
      nextSeq: seq + 1,
    });
  },

  reportLocal: (partial) => {
    const env: ErrorEnvelope = {
      id: genId(),
      ts: nowIso(),
      severity: partial.severity ?? "error",
      source: partial.source ?? "frontend",
      code: partial.code ?? "FRONTEND",
      title: partial.title,
      message: partial.message ?? partial.title,
      scope: partial.scope ?? {},
      details: partial.details ?? {},
      hint: partial.hint ?? null,
    };
    get().ingest(env);
  },

  markRead: (id) => {
    const next = get().entries.map((e) =>
      e.id === id ? { ...e, read: true } : e,
    );
    set({ entries: next, unreadCount: recompUnread(next) });
  },

  markAllRead: () => {
    const next = get().entries.map((e) => ({ ...e, read: true }));
    set({ entries: next, unreadCount: 0 });
  },

  pin: (id, pinned) => {
    const next = get().entries.map((e) => (e.id === id ? { ...e, pinned } : e));
    set({ entries: next });
  },

  dismiss: (id) => {
    const next = get().entries.filter((e) => e.id !== id);
    set({ entries: next, unreadCount: recompUnread(next) });
  },

  clear: () => {
    // Keep pinned entries.
    const next = get().entries.filter((e) => e.pinned);
    set({ entries: next, unreadCount: recompUnread(next) });
  },

  backfill: (events) => {
    // Merge — server is source of truth for `id`, dedup against existing.
    const state = get();
    const existingIds = new Set(state.entries.map((e) => e.id));
    let seq = state.nextSeq;
    const newOnes: ErrorEntry[] = [];
    for (const env of events) {
      if (existingIds.has(env.id)) continue;
      newOnes.push({ ...env, read: false, pinned: false, seq: seq++ });
    }
    if (newOnes.length === 0) return;
    // Sort: server returns oldest-first; we keep newest-first overall.
    const merged = [...newOnes.reverse(), ...state.entries].slice(0, CAP);
    // Re-sort by ts desc to handle interleaving correctly.
    merged.sort((a, b) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0));
    set({
      entries: merged.slice(0, CAP),
      unreadCount: recompUnread(merged),
      nextSeq: seq,
    });
  },
}));
