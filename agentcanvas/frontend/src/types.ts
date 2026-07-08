/** TypeScript interfaces mirroring backend models.py */

export interface AppConfig {
  vlm_max_steps: number;
  slam_backend: string;
  ws_heartbeat_sec: number;
  ignore_loopback_proxy: boolean;
  debug: boolean;
  active_profile: string;
}

export interface ProviderInfo {
  label: string;
  base_url: string;
  api_type: string;
  default_model: string;
  key_env: string;
  key_set: boolean;
  key_source: "file" | "env" | "none";
}

export type ProvidersMap = Record<string, ProviderInfo>;

export interface CapabilityVerdict {
  kind: "locked" | "required" | "range" | "unsupported" | "min_hint";
  value: unknown;
  note: string;
}

export type Capabilities = Record<string, CapabilityVerdict[]>;

export interface LLMProfile {
  provider: string;
  model: string;
  api_key_set: boolean;
  base_url: string;
  api_type: string;
}

export interface ProfilesState {
  active: string;
  profiles: Record<string, LLMProfile>;
}

// ── Navigate types ──

export interface NavPolicyEntry {
  id: string;
  name: string;
  checkpoint: string;
}

export interface NavOrchestrator {
  id: string;
  name: string;
  description: string;
  policies: NavPolicyEntry[];
}

export type NavStatus =
  | "idle"
  | "loading"
  | "running"
  | "paused"
  | "done"
  | "error";

export interface NavStepData {
  step: number;
  action: number;
  action_name: string;
  position: number[];
  orientation: number[];
  rgb_base64: string;
  depth_base64: string;
  done: boolean;
  metrics?: Record<string, number>;
  // State container live previews (home + nodeset-owned), keyed by container id.
  containers?: Record<
    string,
    {
      label: string;
      owner: string;
      states: Record<string, Record<string, unknown>>;
    }
  >;
}

export interface NavLLMStepData {
  step: number;
  type: "reasoning" | "tool_call" | "tool_result" | "decision";
  content: string;
  tool?: string;
  args?: Record<string, unknown>;
  rgb_base64?: string;
  depth_base64?: string;
}

// ── Per-node instance data (for canvas output nodes) ──

/** Last-call LLM usage for one node — from llm_usage WS events (the
 * executor's per-node usage bucket + wall-clock duration). */
export interface LlmUsage {
  calls: number;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  usd_cost: number;
  duration_ms?: number;
}

export interface NodeInstanceData {
  status: NavStatus;
  currentStep: number;
  // Legacy fields (kept for NavigatePage / backward compat)
  currentRgb: string;
  currentDepth: string;
  steps: NavStepData[];
  llmSteps: NavLLMStepData[];
  metrics: Record<string, number> | null;
  /** Per-node display data from viewer_data WS events, keyed by port name. */
  fields: Record<string, unknown>;
  /** Most recent LLM usage for this node (null until it makes a call). */
  usage: LlmUsage | null;
}

// ── Manager types (NodeSets, Envs) ──

export interface NodeSetInfo {
  name: string;
  description: string;
  loaded: boolean;
  mode: "local" | "server";
  requires_server: boolean;
  /** Role bucket (workspace/nodesets/<role>/): env|method|model|policy|common|other. */
  category?: string;
  tools: string[];
  // Declared nodeset-owned container schemas (from the server manifest), so
  // the State panel can show them statically before/without a run.
  containers?: Array<{
    id: string;
    label?: string;
    states?: Record<
      string,
      { type?: string; value_type?: string; lifetime?: string }
    >;
  }>;
}

// ── Saved Graph types ──

export interface SavedGraph {
  _id: string;
  _path: string;
  name: string;
  description: string;
  nodes: unknown[];
  edges: unknown[];
  step_budget: number | null;
  kind?: "graph" | "node";
  group?: string;
  folder?: string; // POSIX subdirectory under the kind root ("" = root)
  presetId?: string;
  containers?: unknown[];
  access_grants?: unknown[];
}

// ── Nodeset env panel panel types ──

export interface EnvPanelField {
  name: string;
  kind: "select" | "number" | "text" | "slider";
  label: string;
  options?: string[] | null;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  placeholder?: string | null;
}

export interface EnvPanelAction {
  name: string;
  label: string;
  side_effect: "run_start" | "run_pause" | "run_stop" | "run_step" | "none";
  enabled_when: "always" | "idle" | "running" | "paused";
}

export interface EnvPanelInfo {
  name: string;
  display_name: string;
  fields: EnvPanelField[];
  actions: EnvPanelAction[];
}

export interface EnvPanelOption {
  value: string | number;
  label: string;
}

/** State payload returned by /api/env-panels/{name}/state — keys depend on
 *  the env panel. We treat it as an open dict and rely on field names from
 *  the schema for rendering. */
export type EnvPanelState = Record<string, unknown>;

export interface EnvPanelActionResult {
  ok: boolean;
  side_effect?: "run_start" | "run_pause" | "run_stop" | "run_step" | "none";
  error?: string;
  [extra: string]: unknown;
}

// ── Eval types ──

export type EvalStatus =
  | "none"
  | "pending"
  | "running"
  | "completed"
  | "cancelled"
  | "error";

export interface EvalEpisodeResult {
  episode_index: number;
  episode_id: string;
  scene_id: string;
  instruction: string;
  metrics: Record<string, number>;
  step_count: number;
  agent_steps: number;
  elapsed_sec: number;
  status: string;
}

export interface EvalRunSummary {
  run_id: string;
  status: EvalStatus;
  episode_count: number;
  total_episodes: number;
  current_episode_index: number;
  completed_count: number;
  error_count: number;
  elapsed_sec: number;
  aggregate_metrics: Record<string, number>;
  agent_id: string | null;
  error: string | null;
}
