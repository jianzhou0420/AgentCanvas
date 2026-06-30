/** Shared hook for subscribing to WebSocket node output data.
 * Used by ViewerLayout to display live execution data. */
import { useFlowStore } from "../../../../useFlowStore";

export function useNodeOutput(id: string) {
  return useFlowStore((s) => s.nodeOutputs[id]);
}
