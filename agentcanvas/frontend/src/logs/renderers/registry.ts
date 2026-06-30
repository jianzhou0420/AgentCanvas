/** Value Type Registry — maps (wire_type | __type | typeof) to React renderer components.
 *
 * Resolution order:
 * 1. If port_wire_types[key] exists → use that wire type
 * 2. If value is object with __type field → use __type
 * 3. typeof primitive → use typeof
 * 4. Fallback → "json"
 */

import type { ComponentType } from "react";
import { lazy } from "react";

export interface ValueRendererProps {
  value: unknown;
  label?: string;
}

// Lazy-loaded renderers for bundle splitting
const ImageRenderer = lazy(() => import("./ImageRenderer"));
const ImageListRenderer = lazy(() => import("./ImageListRenderer"));
const TextRenderer = lazy(() => import("./TextRenderer"));
const ActionRenderer = lazy(() => import("./ActionRenderer"));
const MetricsRenderer = lazy(() => import("./MetricsRenderer"));
const TensorRenderer = lazy(() => import("./TensorRenderer"));
const ErrorRenderer = lazy(() => import("./ErrorRenderer"));
const JsonTreeRenderer = lazy(() => import("./JsonTreeRenderer"));
const PrimitiveRenderer = lazy(() => import("./PrimitiveRenderer"));
const StepResultRenderer = lazy(() => import("./StepResultRenderer"));

const REGISTRY: Record<string, ComponentType<ValueRendererProps>> = {
  // Wire types
  IMAGE: ImageRenderer,
  DEPTH: ImageRenderer,
  OBSERVATION: ImageRenderer, // composite: will show RGB+depth side by side
  "LIST[IMAGE]": ImageListRenderer, // panorama / multi-view: tiled thumbnails
  "LIST[DEPTH]": ImageListRenderer,
  TEXT: TextRenderer,
  ACTION: ActionRenderer,
  METRICS: MetricsRenderer,
  STEP_RESULT: StepResultRenderer,
  BOOL: PrimitiveRenderer,
  STATE: JsonTreeRenderer,
  ANY: JsonTreeRenderer,
  // Asset references from sidecar storage
  asset: ImageRenderer,
  asset_group: ImageRenderer,
  // __type markers from log_serialize()
  image_list: ImageListRenderer, // LIST[IMAGE]/LIST[DEPTH] with per-tile asset refs
  large_string: TextRenderer,
  bytes: TensorRenderer,
  ndarray: TensorRenderer,
  Tensor: TensorRenderer,
  // typeof primitives
  string: PrimitiveRenderer,
  number: PrimitiveRenderer,
  boolean: PrimitiveRenderer,
};

/** Resolve the best renderer for a value. */
export function resolveRenderer(
  value: unknown,
  wireType?: string,
): ComponentType<ValueRendererProps> {
  // 1. Explicit wire type from port_wire_types
  if (wireType && REGISTRY[wireType]) {
    return REGISTRY[wireType];
  }

  // 2. __type marker from log_serialize
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    if (typeof obj.__type === "string" && REGISTRY[obj.__type]) {
      return REGISTRY[obj.__type];
    }
  }

  // 3. typeof primitive
  if (value === null || value === undefined) return PrimitiveRenderer;
  const t = typeof value;
  if (REGISTRY[t]) return REGISTRY[t];

  // 4. Fallback
  return JsonTreeRenderer;
}

export { REGISTRY };
export { ErrorRenderer };
