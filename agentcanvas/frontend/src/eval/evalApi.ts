/** API client for v2 eval endpoints */

const API_BASE = import.meta.env.VITE_API_URL || "";

async function fetchJ<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

async function postJ<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

import type {
  EvalRunSummary,
  EvalEpisodeResult,
  GraphIntrospection,
} from "./types";

export const evalApi = {
  startEval: (data: {
    graph_name: string;
    // Legacy convenience fields — promoted into `selectors` server-side.
    dataset?: string;
    split?: string;
    // Generic cascade (insertion order = env panel field order).
    // Use this for envs whose env panel has fields beyond dataset/split,
    // e.g. SIMPLER's `task_id`, LIBERO's `task_suite`. Do NOT include
    // `episode_index` — the runner pushes that itself.
    selectors?: Record<string, string | number | boolean>;
    // Per-episode selector overrides, parallel to the resolved index list
    // (episode_indices, or [start_episode_index .. +episode_count)). Each
    // entry merges on top of run-level `selectors` for that one episode.
    // Use for cross-task sweeps in a single run (e.g. SIMPLER 25 tasks ×
    // N episodes, ordered task-contiguous so worker subprocesses see
    // few task switches). Do NOT include `episode_index`.
    episode_selectors?: Array<Record<string, string | number | boolean>>;
    episode_indices?: number[];
    episode_count?: number;
    step_budget?: number | null;
    start_episode_index?: number;
    worker_count?: number; // ADR-028: parallel env subprocesses (default 1)
    per_step_budget_sec?: number | null; // ADR-028: per-step timeout; null = nodeset default
  }) => postJ<{ run_id: string }>(`${API_BASE}/api/eval/v2/start`, data),

  stopEval: () => postJ<{ ok: boolean }>(`${API_BASE}/api/eval/v2/stop`, {}),

  getStatus: () =>
    fetchJ<{ status: string; run: EvalRunSummary | null }>(
      `${API_BASE}/api/eval/v2/status`,
    ),

  getEpisodes: () =>
    fetchJ<{ episodes: EvalEpisodeResult[] }>(
      `${API_BASE}/api/eval/v2/episodes`,
    ),

  listRuns: () =>
    fetchJ<{ runs: EvalRunSummary[] }>(`${API_BASE}/api/eval/v2/runs`),

  getRun: (runId: string) =>
    fetchJ<EvalRunSummary>(`${API_BASE}/api/eval/v2/runs/${runId}`),

  deleteRun: (runId: string) =>
    fetch(`${API_BASE}/api/eval/v2/runs/${runId}`, { method: "DELETE" }),

  introspectGraph: (graphName: string) =>
    postJ<GraphIntrospection>(`${API_BASE}/api/eval/v2/introspect`, {
      graph_name: graphName,
    }),

  exportRun: (runId: string) =>
    fetchJ<Record<string, unknown>>(`${API_BASE}/api/eval/v2/export/${runId}`),

  getExecutionMode: () =>
    fetchJ<{ mode: string; holder: string | null }>(
      `${API_BASE}/api/navigate/execution-mode`,
    ),

  listGraphs: () =>
    fetchJ<Array<{ _id: string; name: string; kind?: string }>>(
      `${API_BASE}/api/graphs`,
    ),
};
