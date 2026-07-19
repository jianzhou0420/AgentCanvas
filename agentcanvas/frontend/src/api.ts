/** REST client — pattern from edict api.ts
 *
 * Every non-2xx response and every network failure is reported into the
 * global `useErrorStore` so it surfaces in the bottom-panel Report tab.
 * Callers still get a thrown Error so existing try/catch sites continue
 * to work.
 */

import { useErrorStore } from "./errorStore";
import type { ErrorEnvelope } from "./errors";

const API_BASE = import.meta.env.VITE_API_URL || "";

/** Encode a relative graph id for a `{graph_id:path}` route: encode each
 *  segment but keep the `/` separators literal so subfolder ids match. */
function encPath(id: string): string {
  return id
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
}

/** Internal: turn a non-2xx response into an envelope and ingest. */
async function _reportHttpError(
  url: string,
  method: string,
  res: Response,
): Promise<string> {
  let body: unknown = null;
  let bodyText = "";
  try {
    bodyText = await res.text();
    body = bodyText ? JSON.parse(bodyText) : null;
  } catch {
    /* non-JSON body */
  }
  // If backend returned a structured envelope (from main.py exception handler),
  // use it directly instead of synthesizing one.
  const envelopeFromBody = (body as { error?: ErrorEnvelope })?.error;
  if (envelopeFromBody && envelopeFromBody.id) {
    useErrorStore.getState().ingest(envelopeFromBody);
    return envelopeFromBody.message || `HTTP ${res.status}`;
  }
  const message =
    (body && typeof body === "object" && "detail" in body
      ? String((body as { detail: unknown }).detail)
      : bodyText) || `HTTP ${res.status}`;
  useErrorStore.getState().reportLocal({
    source: "api",
    severity: res.status >= 500 ? "error" : "warning",
    code: `API_${res.status}`,
    title: `${method} ${url} → HTTP ${res.status}`,
    message,
    scope: { endpoint: url, method, status: res.status },
  });
  return message;
}

function _reportNetworkError(url: string, method: string, err: unknown): void {
  useErrorStore.getState().reportLocal({
    source: "api",
    severity: "error",
    code: "API_NETWORK",
    title: `${method} ${url} — network failure`,
    message: err instanceof Error ? err.message : String(err),
    scope: { endpoint: url, method },
  });
}

export async function fetchJ<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch (err) {
    _reportNetworkError(url, "GET", err);
    throw err;
  }
  if (!res.ok) {
    const msg = await _reportHttpError(url, "GET", res);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

async function postJ<T>(
  url: string,
  data?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data !== undefined ? JSON.stringify(data) : undefined,
      signal,
    });
  } catch (err) {
    // A caller-initiated abort is not a network failure — don't report it.
    if ((err as Error)?.name !== "AbortError")
      _reportNetworkError(url, "POST", err);
    throw err;
  }
  if (!res.ok) {
    const msg = await _reportHttpError(url, "POST", res);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

async function deleteJ<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { method: "DELETE" });
  } catch (err) {
    _reportNetworkError(url, "DELETE", err);
    throw err;
  }
  if (!res.ok) {
    const msg = await _reportHttpError(url, "DELETE", res);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

async function putJ<T>(url: string, data: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  } catch (err) {
    _reportNetworkError(url, "PUT", err);
    throw err;
  }
  if (!res.ok) {
    const msg = await _reportHttpError(url, "PUT", res);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

import type {
  AppConfig,
  Capabilities,
  ProfilesState,
  ProvidersMap,
  NodeSetInfo,
  SavedGraph,
  EvalRunSummary,
  EnvPanelInfo,
  EnvPanelState,
  EnvPanelOption,
  EnvPanelActionResult,
} from "./types";

/** GET /api/components/nodesets/{name}/source response. */
export interface NodesetSourceFile {
  name: string;
  mode: "local" | "server";
  requires_server: boolean;
  loaded: boolean;
  is_package: boolean;
  files: string[];
  file: string;
  content: string;
  mtime_ns: number;
  class_name: string | null;
}

/** PUT /api/components/nodesets/{name}/source outcome (discriminated on `kind`). */
export type SaveSourceResult =
  | { ok: true; mtime_ns: number; mode: "local" | "server"; stale: boolean; run_active: boolean }
  | { ok: false; kind: "syntax"; msg: string; line: number | null; offset: number | null }
  | { ok: false; kind: "conflict" }
  | { ok: false; kind: "error"; message: string };

/** One editable slice of a nodeset file (Source tab): globals block,
 *  referenced function, or the node's class. 1-based inclusive lines. */
export interface SourceSegment {
  kind: "globals" | "function" | "class";
  name: string;
  start_line: number;
  end_line: number;
  text: string;
}

/** GET /api/components/nodesets/{name}/source/scoped response. */
export interface ScopedSource {
  name: string;
  node_type: string;
  mode: "local" | "server";
  requires_server: boolean;
  loaded: boolean;
  file: string;
  mtime_ns: number;
  segments: SourceSegment[];
}

export type SaveScopedResult =
  | {
      ok: true;
      mtime_ns: number;
      mode: "local" | "server";
      stale: boolean;
      run_active: boolean;
      segments: SourceSegment[] | null;
    }
  | { ok: false; kind: "syntax"; msg: string; line: number | null; offset: number | null }
  | { ok: false; kind: "conflict" }
  | { ok: false; kind: "error"; message: string };

export const api = {
  // Config
  getConfig: () => fetchJ<AppConfig>(`${API_BASE}/api/config/`),
  updateConfig: (updates: Record<string, unknown>) =>
    putJ<{ ok: boolean; changed: Record<string, unknown> }>(
      `${API_BASE}/api/config/`,
      updates,
    ),

  // Profiles
  getProfiles: () => fetchJ<ProfilesState>(`${API_BASE}/api/profiles/`),
  createProfile: (data: {
    name: string;
    provider: string;
    model: string;
    base_url?: string;
    api_type?: string;
  }) => postJ<{ ok: boolean; name: string }>(`${API_BASE}/api/profiles/`, data),
  updateProfile: (name: string, data: Record<string, unknown>) =>
    putJ<{ ok: boolean }>(
      `${API_BASE}/api/profiles/${encodeURIComponent(name)}`,
      data,
    ),
  deleteProfile: (name: string) =>
    deleteJ<{ ok: boolean }>(
      `${API_BASE}/api/profiles/${encodeURIComponent(name)}`,
    ),
  activateProfile: (name: string) =>
    postJ<{ ok: boolean; active: string }>(
      `${API_BASE}/api/profiles/activate`,
      { name },
    ),
  // Providers — registry, keys (~/.agentcanvas/.keys), models, rulebook
  getProviders: () => fetchJ<ProvidersMap>(`${API_BASE}/api/providers/`),
  setProviderKey: (provider: string, key: string) =>
    putJ<{ ok: boolean; key_env: string; key_source: string }>(
      `${API_BASE}/api/providers/${encodeURIComponent(provider)}/key`,
      { key },
    ),
  deleteProviderKey: (provider: string) =>
    deleteJ<{ ok: boolean; removed: boolean; key_source: string }>(
      `${API_BASE}/api/providers/${encodeURIComponent(provider)}/key`,
    ),
  validateProviderKey: (provider: string) =>
    postJ<{ ok: boolean; message: string }>(
      `${API_BASE}/api/providers/${encodeURIComponent(provider)}/validate`,
      {},
    ),
  getProviderModels: (provider: string) =>
    fetchJ<{ models: string[]; provider: string }>(
      `${API_BASE}/api/providers/${encodeURIComponent(provider)}/models`,
    ),
  getProviderCapabilities: (provider: string, model: string) =>
    fetchJ<{ provider: string; model: string; capabilities: Capabilities }>(
      `${API_BASE}/api/providers/${encodeURIComponent(provider)}/capabilities?model=${encodeURIComponent(model)}`,
    ),

  // Eval — status poll (graph-driven runs live under /api/eval/v2; the
  // start/stop/episodes/export surface lives in eval/evalApi.ts)
  getEvalStatus: () =>
    fetchJ<{ status: string; run: EvalRunSummary | null }>(
      `${API_BASE}/api/eval/v2/status`,
    ),

  // Navigate — Discovery
  getNavPolicies: () =>
    fetchJ<{ id: string; name: string; checkpoint: string; config: string }[]>(
      `${API_BASE}/api/navigate/policies`,
    ),

  // Components — Node schema discovery
  getNodeSchemas: () =>
    fetchJ<Record<string, unknown>[]>(
      `${API_BASE}/api/components/node-schemas`,
    ),

  // Navigate — Loop execution
  navRun: (data: {
    loop_definition: Record<string, unknown>;
    execution_id?: string;
    step_delay_ms?: number;
  }) =>
    postJ<{ ok: boolean; execution_id?: string }>(
      `${API_BASE}/api/navigate/run`,
      data,
    ),
  navRunPause: () =>
    postJ<{ ok: boolean }>(`${API_BASE}/api/navigate/run/pause`, {}),
  navRunStop: () =>
    postJ<{ ok: boolean }>(`${API_BASE}/api/navigate/run/stop`, {}),
  navRunStatus: () =>
    fetchJ<Record<string, unknown>>(`${API_BASE}/api/navigate/run/status`),

  // Server mode
  listServers: () =>
    fetchJ<ServerStatus[]>(`${API_BASE}/api/components/servers`),
  startServer: (name: string) =>
    postJ<ServerStatus>(`${API_BASE}/api/components/servers/${name}/start`, {}),
  stopServer: (name: string) =>
    postJ<ServerStatus>(`${API_BASE}/api/components/servers/${name}/stop`, {}),
  restartServer: (name: string) =>
    postJ<ServerStatus>(
      `${API_BASE}/api/components/servers/${name}/restart`,
      {},
    ),

  // Component management (NodeSet Manager page)
  listComponents: () =>
    fetchJ<Record<string, string[]>>(`${API_BASE}/api/components/`),
  reloadComponents: () =>
    postJ<{ ok: boolean; components: Record<string, number> }>(
      `${API_BASE}/api/components/reload`,
      {},
    ),
  listNodesets: () =>
    fetchJ<NodeSetInfo[]>(`${API_BASE}/api/components/nodesets`),
  getNodesetSource: (name: string, file?: string, nodeType?: string) => {
    const q = new URLSearchParams();
    if (file) q.set("file", file);
    if (nodeType) q.set("node_type", nodeType);
    const qs = q.toString();
    return fetchJ<NodesetSourceFile>(
      `${API_BASE}/api/components/nodesets/${encodeURIComponent(name)}/source${qs ? `?${qs}` : ""}`,
    );
  },
  getScopedSource: (name: string, nodeType: string) =>
    fetchJ<ScopedSource>(
      `${API_BASE}/api/components/nodesets/${encodeURIComponent(name)}/source/scoped?node_type=${encodeURIComponent(nodeType)}`,
    ),
  saveScopedSource: async (
    name: string,
    body: {
      file: string;
      node_type: string;
      base_mtime_ns: number | null;
      segments: { start_line: number; end_line: number; text: string }[];
    },
  ): Promise<SaveScopedResult> => {
    let res: Response;
    try {
      res = await fetch(
        `${API_BASE}/api/components/nodesets/${encodeURIComponent(name)}/source/scoped`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
    } catch (err) {
      return {
        ok: false,
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      };
    }
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      /* empty / non-JSON body */
    }
    if (res.ok) return parsed as SaveScopedResult;
    const detail = (parsed as { detail?: unknown } | null)?.detail;
    if (
      res.status === 400 &&
      detail &&
      typeof detail === "object" &&
      (detail as { error?: string }).error === "syntax"
    ) {
      const s = detail as { msg: string; line: number | null; offset: number | null };
      return { ok: false, kind: "syntax", msg: s.msg, line: s.line, offset: s.offset };
    }
    if (res.status === 409) return { ok: false, kind: "conflict" };
    return {
      ok: false,
      kind: "error",
      message: typeof detail === "string" ? detail : `HTTP ${res.status}`,
    };
  },
  // Deliberately not putJ: 400 (syntax) / 409 (disk conflict) are expected
  // editor outcomes handled inline in the source drawer — they must not
  // land in the Report tab as app errors.
  saveNodesetSource: async (
    name: string,
    body: { file: string; content: string; base_mtime_ns: number | null },
  ): Promise<SaveSourceResult> => {
    let res: Response;
    try {
      res = await fetch(
        `${API_BASE}/api/components/nodesets/${encodeURIComponent(name)}/source`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
    } catch (err) {
      return {
        ok: false,
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      };
    }
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      /* empty / non-JSON body */
    }
    if (res.ok) return parsed as SaveSourceResult;
    const detail = (parsed as { detail?: unknown } | null)?.detail;
    if (
      res.status === 400 &&
      detail &&
      typeof detail === "object" &&
      (detail as { error?: string }).error === "syntax"
    ) {
      const s = detail as { msg: string; line: number | null; offset: number | null };
      return { ok: false, kind: "syntax", msg: s.msg, line: s.line, offset: s.offset };
    }
    if (res.status === 409) return { ok: false, kind: "conflict" };
    return {
      ok: false,
      kind: "error",
      message: typeof detail === "string" ? detail : `HTTP ${res.status}`,
    };
  },
  loadNodeset: (name: string) =>
    postJ<{ ok: boolean; tools: string[] }>(
      `${API_BASE}/api/components/nodesets/${name}/load`,
      {},
    ),
  unloadNodeset: (name: string) =>
    postJ<{ ok: boolean }>(
      `${API_BASE}/api/components/nodesets/${name}/unload`,
      {},
    ),
  ensureNodesets: (nodeTypes: string[], signal?: AbortSignal) =>
    postJ<{
      loaded: string[];
      already_loaded: string[];
      failed: string[];
      unknown: string[];
    }>(
      `${API_BASE}/api/components/nodesets/ensure`,
      { node_types: nodeTypes },
      signal,
    ),

  // Graphs (saved graph definitions)
  listGraphs: () => fetchJ<SavedGraph[]>(`${API_BASE}/api/graphs`),
  getGraph: (id: string) =>
    fetchJ<SavedGraph>(`${API_BASE}/api/graphs/${encPath(id)}`),
  saveGraph: (data: {
    name: string;
    description?: string;
    nodes: unknown[];
    edges: unknown[];
    containers?: unknown[];
    access_grants?: unknown[];
    step_budget?: number | null;
    eval_graph?: boolean;
    kind?: "graph" | "node";
    group?: string;
    folder?: string;
    presetId?: string;
  }) => postJ<{ id: string; path: string }>(`${API_BASE}/api/graphs`, data),
  updateGraph: (
    id: string,
    data: {
      name: string;
      description?: string;
      nodes: unknown[];
      edges: unknown[];
      containers?: unknown[];
      access_grants?: unknown[];
      step_budget?: number | null;
      eval_graph?: boolean;
    },
  ) =>
    putJ<{ id: string; path: string }>(
      `${API_BASE}/api/graphs/${encPath(id)}`,
      data,
    ),
  deleteGraph: (id: string) =>
    deleteJ<{ deleted: string }>(`${API_BASE}/api/graphs/${encPath(id)}`),
  moveGraph: (id: string, destFolder: string, newName?: string) =>
    postJ<{ id: string; path: string }>(
      `${API_BASE}/api/graphs/${encPath(id)}/move`,
      { dest_folder: destFolder, new_name: newName ?? null },
    ),

  // Graph folders (real subdirectories under the kind root)
  listGraphFolders: (kind: "graph" | "node") =>
    fetchJ<string[]>(
      `${API_BASE}/api/graphs/folders?kind=${encodeURIComponent(kind)}`,
    ),
  createGraphFolder: (kind: "graph" | "node", path: string) =>
    postJ<{ created: string }>(`${API_BASE}/api/graphs/folders`, {
      kind,
      path,
    }),
  renameGraphFolder: (kind: "graph" | "node", path: string, newPath: string) =>
    postJ<{ path: string }>(`${API_BASE}/api/graphs/folders/rename`, {
      kind,
      path,
      new_path: newPath,
    }),
  deleteGraphFolder: (
    kind: "graph" | "node",
    path: string,
    recursive = false,
  ) =>
    deleteJ<{ deleted: string }>(
      `${API_BASE}/api/graphs/folders?kind=${encodeURIComponent(kind)}&path=${encodeURIComponent(path)}&recursive=${recursive}`,
    ),

  // Layout. `dimensions` carries canvas-measured node sizes so columns are
  // spaced by real width and rows by real height (avoids overlap on wide nodes).
  layoutGraph: (
    graph: Record<string, unknown>,
    dimensions?: Record<string, { width: number; height: number }>,
  ) =>
    postJ<SavedGraph>(`${API_BASE}/api/graphs/layout`, { ...graph, dimensions }),

  // Nodeset env panels (canvas control panel — replaces /api/env)
  envPanelList: () => fetchJ<EnvPanelInfo[]>(`${API_BASE}/api/env-panels`),
  envPanelState: (name: string) =>
    fetchJ<EnvPanelState>(
      `${API_BASE}/api/env-panels/${encodeURIComponent(name)}/state`,
    ),
  envPanelOptions: (name: string, field: string) =>
    fetchJ<EnvPanelOption[]>(
      `${API_BASE}/api/env-panels/${encodeURIComponent(name)}/options/${encodeURIComponent(field)}`,
    ),
  envPanelSetField: (name: string, field: string, value: unknown) =>
    postJ<EnvPanelState>(
      `${API_BASE}/api/env-panels/${encodeURIComponent(name)}/field/${encodeURIComponent(field)}`,
      { value },
    ),
  envPanelAction: (
    name: string,
    action: string,
    params: Record<string, unknown> = {},
  ) =>
    postJ<EnvPanelActionResult>(
      `${API_BASE}/api/env-panels/${encodeURIComponent(name)}/action/${encodeURIComponent(action)}`,
      { params },
    ),
};

export interface ServerStatus {
  name: string;
  description: string;
  url: string;
  status: "stopped" | "starting" | "connected" | "unreachable" | "error";
  pid: number | null;
  connected: boolean;
  error: string | null;
  auto_restart: boolean;
  nodes?: string[];
}
