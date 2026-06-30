/** Bridge to avoid circular import between store.ts and useFlowStore.ts.
 *  useFlowStore registers itself here at creation time.
 *  store.ts reads from here in WS callbacks (lazy, never at import time). */

import type { NodeInstanceData } from "../types";

interface FlowStoreBridge {
  getActiveExecution: (
    executionId: string,
  ) => { agentNodeId: string; outputNodeIds: string[] } | undefined;
  updateNodeOutput: (nodeId: string, update: Partial<NodeInstanceData>) => void;
  getNodeOutput: (nodeId: string) => NodeInstanceData | undefined;
  setContainersLive: (
    payload: Record<
      string,
      {
        label: string;
        owner: string;
        states: Record<string, Record<string, unknown>>;
      }
    > | null,
  ) => void;
}

let _bridge: FlowStoreBridge | null = null;

export function registerFlowStoreBridge(bridge: FlowStoreBridge) {
  _bridge = bridge;
}

export function getFlowStoreBridge(): FlowStoreBridge | null {
  return _bridge;
}
