/** Error envelope schema — mirror of backend `agentcanvas/backend/app/errors.py`. */

export type ErrorSeverity = "error" | "warning" | "info" | "debug";

export type ErrorSource =
  | "node"
  | "graph"
  | "api"
  | "ws"
  | "frontend"
  | "plugin"
  | "eval"
  | "log";

export interface ErrorEnvelope {
  id: string;
  ts: string; // ISO-8601
  severity: ErrorSeverity;
  source: ErrorSource;
  code: string;
  title: string;
  message: string;
  scope: Record<string, unknown>;
  details: Record<string, unknown>;
  hint: string | null;
}
