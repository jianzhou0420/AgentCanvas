/** API client for /api/replay/* — log replay timeline endpoints. */

const API_BASE = import.meta.env.VITE_API_URL || "";

async function fetchJ<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface ReplayStep {
  step_index: number;
  frame_url: string;
  info: Record<string, unknown>;
  render_params: Record<string, unknown> | null;
}

export interface ReplayEpisode {
  run_id: string;
  /** Composite id `{run_id}_ep{idx:04d}` — used to fetch frame assets
   *  via /api/logs/{execution_id}/assets/{filename}. */
  execution_id: string;
  episode_index: number;
  episode_id: string;
  instruction: string;
  step_count: number;
  steps: ReplayStep[];
  metrics: Record<string, number>;
  supports_smooth: boolean;
  scene_id: string;
}

export interface ReplayEpisodeSummary {
  episode_index: number;
  episode_id: string;
  instruction: string;
  /** eval_storage's authoritative step count (each episode owns its own
   *  log dir now, so there is no parser re-derivation that can disagree). */
  step_count: number;
  metrics: Record<string, number>;
  status: string;
  scene_id: string;
  /** Whether this episode's log.jsonl is present on disk (a crashed or
   *  aborted episode may have none). */
  has_log: boolean;
}

export interface ReplayEpisodesResponse {
  run_id: string;
  env_nodeset: string | null;
  graph_name: string | null;
  episodes: ReplayEpisodeSummary[];
}

export const replayApi = {
  listEpisodes: (runId: string) =>
    fetchJ<ReplayEpisodesResponse>(`${API_BASE}/api/replay/${runId}/episodes`),

  getEpisode: (runId: string, episodeIndex: number) =>
    fetchJ<ReplayEpisode>(
      `${API_BASE}/api/replay/${runId}/episode/${episodeIndex}`,
    ),

  /** Build a fully-qualified URL for a step's frame asset.
   * `executionId` is the episode's composite id `{run_id}_ep{idx:04d}`. */
  assetUrl: (executionId: string, framePath: string) =>
    `${API_BASE}/api/logs/${executionId}/assets/${framePath.replace(/^assets\//, "")}`,

  /** Build a smooth-mode interpolated frame URL.
   * t ∈ [0, 1]: 0 = pose at stepIndex, 1 = pose at stepIndex+1.
   */
  smoothFrameUrl: (
    runId: string,
    episodeIndex: number,
    stepIndex: number,
    t: number,
  ) =>
    `${API_BASE}/api/replay/${runId}/episode/${episodeIndex}/step/${stepIndex}/frame?t=${t}`,
};
