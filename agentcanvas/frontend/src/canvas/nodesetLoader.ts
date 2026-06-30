/** Auto-load nodesets required by a graph.
 *
 * Delegates to the backend POST /api/components/nodesets/ensure endpoint,
 * which uses ComponentRegistry.ensure_nodesets_for_graph() — the same
 * logic used by the eval batch runner.
 */

import { api } from "../api";

/** Extract nodeset name from a node type. Returns null for built-in types. */
export function parseNodesetName(nodeType: string): string | null {
  const idx = nodeType.indexOf("__");
  return idx > 0 ? nodeType.slice(0, idx) : null;
}

export interface LoadResult {
  loaded: string[]; // nodesets that were loaded
  already_loaded: string[]; // nodesets that were already active (note: snake_case from backend)
  alreadyLoaded: string[]; // alias for compatibility
  failed: string[]; // nodesets that failed to load
  unknown: string[]; // nodeset names not found in discovered list
}

/**
 * Ensure all nodesets required by a graph's node types are loaded.
 *
 * Calls the shared backend endpoint that handles detection + loading.
 *
 * @param nodeTypes - Array of node type strings from the graph
 * @returns LoadResult with details of what happened
 */
export async function ensureNodesetsLoaded(
  nodeTypes: string[],
  signal?: AbortSignal,
): Promise<LoadResult> {
  // Filter to only nodeset node types (contain '__')
  const nodesetTypes = nodeTypes.filter((nt) => nt.includes("__"));
  if (nodesetTypes.length === 0) {
    return {
      loaded: [],
      already_loaded: [],
      alreadyLoaded: [],
      failed: [],
      unknown: [],
    };
  }

  try {
    const result = await api.ensureNodesets(nodesetTypes, signal);
    return {
      ...result,
      // Provide camelCase alias for backward compat
      alreadyLoaded: result.already_loaded ?? [],
    };
  } catch (err) {
    // A caller cancel must propagate so the load flow can bail cleanly
    // instead of reporting every nodeset as "failed".
    if ((err as Error)?.name === "AbortError") throw err;
    // Fallback: extract unique prefixes and report all as failed
    const prefixes = [...new Set(nodesetTypes.map((nt) => nt.split("__")[0]))];
    return {
      loaded: [],
      already_loaded: [],
      alreadyLoaded: [],
      failed: prefixes,
      unknown: [],
    };
  }
}
