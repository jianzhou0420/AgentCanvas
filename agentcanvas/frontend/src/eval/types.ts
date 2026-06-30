/** Eval-specific types for the v2 eval page */

export type EvalStatus =
  | "none"
  | "pending"
  | "running"
  | "completed"
  | "cancelled"
  | "error";

export interface EvalRunSummary {
  run_id: string;
  graph_name: string;
  env_nodeset: string;
  status: EvalStatus;
  // Generic cascade — keys + values that were pushed through the env
  // env panel. Insertion order = the order in which the runner pushed
  // them. Empty for runs that didn't use the env panel cascade path.
  selectors: Record<string, string | number | boolean>;
  // Legacy convenience fields — kept for back-compat with old summaries.
  // For new code, prefer reading `selectors`.
  split: string;
  episode_count: number;
  total_episodes: number;
  completed_count: number;
  error_count: number;
  elapsed_sec: number;
  aggregate_metrics: Record<string, number>;
  // Per-task breakdown of metric means, grouped by task_id (or canonical
  // selectors JSON) — populated for cross-task sweeps. Optional for
  // back-compat with summaries persisted before the per-task grouping.
  aggregate_by_task?: Record<string, Record<string, number>>;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface EvalEpisodeResult {
  run_id: string;
  episode_index: number;
  episode_id: string;
  scene_id: string;
  instruction: string;
  metrics: Record<string, number>;
  step_count: number;
  elapsed_sec: number;
  status: string;
  error: string | null;
  // ADR-028 PB-3: which pool worker drove this episode. 0 at worker_count=1.
  // Optional for back-compat with runs persisted before PB-3.
  worker_id?: number;
  // Effective selectors pushed through the env panel for this
  // episode (run-level merged with per-episode override). Optional for
  // back-compat with summaries persisted before the cross-task path.
  selectors?: Record<string, string | number | boolean>;
}

export interface GraphIntrospection {
  graph_name: string;
  env_nodeset: string | null;
  loaded: boolean;
  metadata: EvalMetadata | null;
}

export interface EvalMetadata {
  env_name: string;
  splits: string[];
  episode_counts: Record<string, number>;
  metrics: string[];
  supports_set_episode: boolean;
  step_budget: number;
}
