/** Pipeline execution — sends GraphDefinition JSON to backend /run endpoint.
 *
 * In the unified canvas, the root graph IS the execution graph.
 */

import { useStore } from "../store";
import { api } from "../api";
import type { GraphDefinition } from "./types";

/** Run a graph definition on the backend. */
export async function runPipeline(
  graph: GraphDefinition,
  executionId?: string,
): Promise<void> {
  const store = useStore.getState();

  // Pre-flight: eval graphs must declare at least one graphOut node.
  // The backend's validate_graph_connectivity enforces this on submit, but
  // catching it client-side surfaces a clearer error before the network round-trip.
  if (graph.eval_graph !== false) {
    const graphOutCount = graph.nodes.filter(
      (n) => n.type === "graphOut",
    ).length;
    if (graphOutCount < 1) {
      throw new Error(
        `Eval graph must declare at least one graphOut node, found 0. ` +
          `Set 'eval_graph: false' on the graph if it does not produce metrics.`,
      );
    }
  }

  const loopDef = {
    name: graph.name,
    description: graph.description,
    nodes: graph.nodes,
    edges: graph.edges,
    containers: graph.containers ?? [],
    access_grants: graph.access_grants ?? [],
    step_budget: graph.step_budget ?? 500,
    eval_graph: graph.eval_graph ?? true,
    presetId: graph.presetId,
    hooks: graph.hooks ?? [],
  };

  await api.navRun({
    loop_definition: loopDef as unknown as Record<string, unknown>,
    execution_id: executionId,
    step_delay_ms: store.navStepDelay,
  });
}
