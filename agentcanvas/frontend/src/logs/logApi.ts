/** Log REST API client. */

import type { LogEntry, ExecutionListItem, ExecutionSummary } from "./types";
import { fetchJ } from "../api";

const API = import.meta.env.VITE_API_URL || "";

interface EntriesResponse {
  execution_id: string;
  entries: LogEntry[];
  total: number;
  source: string;
}

export const logApi = {
  listExecutions: () =>
    fetchJ<{ executions: ExecutionListItem[] }>(`${API}/api/logs`),

  getEntries: (
    executionId: string,
    params?: {
      node_id?: string;
      node_type?: string;
      step?: number;
      limit?: number;
      offset?: number;
    },
  ) => {
    const sp = new URLSearchParams();
    if (params?.node_id) sp.set("node_id", params.node_id);
    if (params?.node_type) sp.set("node_type", params.node_type);
    if (params?.step !== undefined) sp.set("step", String(params.step));
    if (params?.limit) sp.set("limit", String(params.limit));
    if (params?.offset) sp.set("offset", String(params.offset));
    const qs = sp.toString();
    return fetchJ<EntriesResponse>(
      `${API}/api/logs/${executionId}${qs ? "?" + qs : ""}`,
    );
  },

  getNodeHistory: (executionId: string, nodeId: string, limit = 100) =>
    fetchJ<EntriesResponse>(
      `${API}/api/logs/${executionId}/node/${nodeId}?limit=${limit}`,
    ),

  getSummary: (executionId: string) =>
    fetchJ<ExecutionSummary>(`${API}/api/logs/${executionId}/summary`),

  getGraph: (executionId: string) =>
    fetchJ<Record<string, unknown>>(`${API}/api/logs/${executionId}/graph`),

  assetUrl: (executionId: string, assetPath: string): string => {
    // assetPath is like "assets/s3_f12_env__rgb.jpg" — strip "assets/" prefix
    const filename = assetPath.replace(/^assets\//, "");
    return `${API}/api/logs/${executionId}/assets/${filename}`;
  },
};
