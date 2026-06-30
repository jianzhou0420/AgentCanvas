/** Display type registry for viewer field renderers.
 *
 * Maps display_type strings to React components. Follows the same
 * registry pattern as logs/renderers/registry.ts.
 *
 * To add a new display type:
 * 1. Create fields/MyField.tsx implementing DisplayFieldProps
 * 2. Add a lazy import + entry to REGISTRY below
 */

import type { ComponentType } from "react";
import { lazy } from "react";

/** Standardized props for all display field renderers. */
export interface DisplayFieldProps {
  /** The resolved display value for this field. */
  value: unknown;
  /** Human-readable label. */
  label: string;
  /** Max entries to show (for list-type fields). */
  max_visible?: number;
  /** Renderer-specific options. */
  options?: Record<string, unknown>;
}

// Lazy-loaded renderers for bundle splitting
const ImageViewerField = lazy(() => import("./fields/ImageViewerField"));
const LogListField = lazy(() => import("./fields/LogListField"));
const MetricTableField = lazy(() => import("./fields/MetricTableField"));
const TextViewerField = lazy(() => import("./fields/TextViewerField"));

const REGISTRY: Record<string, ComponentType<DisplayFieldProps>> = {
  image_viewer: ImageViewerField,
  log_list: LogListField,
  metric_table: MetricTableField,
  text_viewer: TextViewerField,
};

/** Resolve a display field renderer by type name. Returns null for unknown types. */
export function resolveFieldRenderer(
  displayType: string,
): ComponentType<DisplayFieldProps> | null {
  return REGISTRY[displayType] || null;
}

export { REGISTRY };
