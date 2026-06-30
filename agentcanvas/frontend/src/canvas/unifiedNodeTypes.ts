/** Unified node type registry — maps node type strings to React components.
 *
 * Only nodes with CUSTOM UI (composite nav, state containers) need explicit
 * entries here. All other nodes — including output viewers — are rendered by
 * GenericBlockRenderer via the Proxy fallback. Their UI is driven entirely by
 * Python's BaseCanvasNode.ui_config (ADR-007, ADR-019).
 */

import type { NodeTypes } from "@xyflow/react";

import GenericBlockRenderer from "./nodes/agentloop/inner/GenericBlockRenderer";
import CompositeCanvasNode from "./nodes/composite/CompositeCanvasNode";
import StateContainerNode from "./nodes/state/StateContainerNode";

export const unifiedNodeTypes = {
  compositeNode: CompositeCanvasNode,
  stateContainer: StateContainerNode,
  _generic: GenericBlockRenderer,
} as const;

/**
 * Proxied node types — all node types NOT listed above (llmCall, iterIn,
 * habitat_step, imageViewer, textViewer, textScroll, actionLog, metrics, etc.)
 * fall back to GenericBlockRenderer which reads their ui_config from
 * data._schema. This is the core of the Python-driven node system.
 */
export const proxiedNodeTypes = new Proxy(
  unifiedNodeTypes as Record<string, unknown>,
  {
    get(target, prop) {
      if (typeof prop === "string" && prop in target) return target[prop];
      if (typeof prop === "string") return GenericBlockRenderer;
      return undefined;
    },
  },
) as NodeTypes;

/** Set of node types that are frontend-only (not sent to backend for execution).
 *  stateContainer nodes are extracted separately as ContainerDefs. */
export const FRONTEND_ONLY_TYPES = new Set<string>(["stateContainer"]);

/** Legacy set of known output viewer types (fallback when schema unavailable). */
export const OUTPUT_NODE_TYPES = new Set([
  "imageViewer",
  "textViewer",
  "textScroll",
  "actionLog",
  "metrics",
]);

/** Annotation node types — free-floating canvas commentary, never participate
 *  in execution. Mirrors backend `layout.py:ANNOTATION_TYPES`. Add new
 *  annotation forms (sticker, link, etc.) here as they ship. */
export const ANNOTATION_NODE_TYPES = new Set<string>(["note"]);

/** Schema-driven check: is this node an output viewer? */
export function isOutputNode(
  nodeData: Record<string, unknown> | undefined,
): boolean {
  const schema = nodeData?._schema as Record<string, unknown> | undefined;
  if (schema) {
    const uiConfig = schema.ui_config as Record<string, unknown> | undefined;
    if (uiConfig?.layout === "viewer") return true;
    if (schema.category === "output") return true;
  }
  return false;
}
