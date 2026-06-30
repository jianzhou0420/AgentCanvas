/** Log system types — canonical definitions for the Log Viewer page. */

export interface LogEntry {
  timestamp: string;
  execution_id: string;
  source: "canvas" | "eval";
  step: number;
  node_id: string;
  node_type: string;
  node_label: string;
  duration_ms: number;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  inner_log: { key: string; value: unknown }[];
  port_wire_types: Record<string, string>;
  error: string | null;
}

export interface ExecutionListItem {
  execution_id: string;
  source: "canvas" | "eval";
  size_bytes: number;
  modified: number;
}

export interface ExecutionSummary {
  execution_id: string;
  source: string;
  started_at: string | null;
  ended_at: string | null;
  total_steps: number;
  total_firings: number;
  error_count: number;
  node_types_fired: string[];
}

/** Asset file reference from sidecar storage. */
export interface AssetRef {
  __type: "asset";
  path: string;
  wire_type: string;
  shape?: number[];
  dtype?: string;
}

/** Composite asset group (e.g., OBSERVATION with rgb + depth). */
export interface AssetGroupRef {
  __type: "asset_group";
  wire_type: string;
  rgb?: AssetRef;
  depth?: AssetRef;
}

/** Lightweight WS exec_log event (no inputs/outputs). */
export interface ExecLogEvent {
  step: number;
  node_id: string;
  node_type: string;
  node_label: string;
  duration_ms: number;
  error: string | null;
  has_inner_log: boolean;
}
