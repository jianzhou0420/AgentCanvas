/** Shared types, color maps, and utilities for GenericBlockRenderer layouts. */

// ── Schema interfaces (mirror Python dataclasses) ──

export interface PortSchema {
  name: string;
  wire_type: string;
  description: string;
  optional: boolean;
  /** iterIn only: per-port persist flag (value holds across fires when true). */
  persist?: boolean;
  /** iterIn only: which writer declared this port — drives banded rendering. */
  origin?: "init" | "iterOut";
  /** iterIn only: original (unprefixed) port name on the writer node. */
  writer_name?: string;
}

export interface ConfigFieldSchema {
  name: string;
  field_type: string; // "label" | "slider" | "text" | "select" | "toggle" | "textarea" | "port_list"
  label: string;
  default: unknown;
  min: number | null;
  max: number | null;
  step: number | null;
  options: Array<{ value: string; label: string }> | null;
  placeholder: string;
  /** For field_type "port_list": which port array to control ("input_ports" | "output_ports"). */
  port_side?: string;
  /** For field_type "port_list" on iterIn: render a per-port persist checkbox column. */
  show_persist_toggle?: boolean;
  /** Properties-panel group ("model" | "prompt" | "wiring"); "" = ungrouped. */
  section?: string;
  /** Render inline on the canvas card (false: panel-only). Default true. */
  on_card?: boolean;
  /** Two-state fields: placeholder shown while unset (value not sent). */
  unset_label?: string;
}

export interface DisplayFieldSchema {
  name: string;
  display_type: string; // "image_viewer" | "log_list" | "metric_table" | "text_viewer"
  label: string;
  data_key: string;
  max_visible: number;
  accumulate?: boolean; // true = append to array; false = replace
}

export interface UIConfigSchema {
  color: string;
  layout: string;
  width: string;
  min_width: string;
  max_width: string;
  min_height: string;
  rounding: string;
  config_fields: ConfigFieldSchema[];
  display_fields: DisplayFieldSchema[];
}

export interface NodeSchema {
  type: string;
  display_name: string;
  description: string;
  category: string;
  icon: string;
  kind: string;
  config_schema: Record<string, unknown>;
  default_config: Record<string, unknown>;
  ui_config?: UIConfigSchema;
  input_ports: PortSchema[];
  output_ports: PortSchema[];
}

// ── Wire-type handle colours ──

export const WIRE_COLORS: Record<string, string> = {
  IMAGE: "#22c55e",
  DEPTH: "#06b6d4",
  ACTION: "#f59e0b",
  POSE: "#8b5cf6",
  TEXT: "#6b7280",
  BOOL: "#ef4444",
  METRICS: "#3b82f6",
  OBSERVATION: "#22c55e",
  STEP_RESULT: "#ec4899",
  ANY: "#9ca3af",
};

// ── LIST[T] modifier helpers (ADR-027) ──

export function isListWireType(wireType: string): boolean {
  return wireType.startsWith("LIST[") && wireType.endsWith("]");
}

export function unwrapListWireType(wireType: string): string {
  return isListWireType(wireType) ? wireType.slice(5, -1) : wireType;
}

/** Look up the display color for any wire type, including LIST[T]. */
export function getWireColor(wireType: string): string {
  const inner = unwrapListWireType(wireType);
  return WIRE_COLORS[inner] ?? WIRE_COLORS.ANY;
}

/** Edge-drag compatibility check (ADR-027).
 *
 * Rules:
 * - ANY on either side → allowed (escape hatch).
 * - Equal types → allowed.
 * - T → LIST[T] → allowed (consumer-side auto-wrap).
 * - LIST[T] → T → rejected (lossy).
 * - Different inner types → rejected.
 * Missing/unknown types default to allowed.
 */
export function isCompatibleWireConnection(
  sourceType: string | undefined,
  targetType: string | undefined,
): boolean {
  if (!sourceType || !targetType) return true;
  if (sourceType === "ANY" || targetType === "ANY") return true;
  if (sourceType === targetType) return true;
  const srcList = isListWireType(sourceType);
  const tgtList = isListWireType(targetType);
  if (!srcList && tgtList) {
    return sourceType === unwrapListWireType(targetType);
  }
  if (srcList && !tgtList) return false;
  return false;
}

// ── Colour map (Python color key → Tailwind classes) ──

export interface ColorSet {
  border: string;
  bg: string;
  text: string;
  handle: string;
}

export const COLOR_MAP: Record<string, ColorSet> = {
  yellow: {
    border: "border-yellow-500",
    bg: "bg-yellow-900",
    text: "text-yellow-200",
    handle: "#eab308",
  },
  amber: {
    border: "border-amber-500",
    bg: "bg-amber-900",
    text: "text-amber-200",
    handle: "#f59e0b",
  },
  emerald: {
    border: "border-emerald-500",
    bg: "bg-emerald-900/80",
    text: "text-emerald-300",
    handle: "#34d399",
  },
  pink: {
    border: "border-pink-500",
    bg: "bg-pink-900",
    text: "text-pink-200",
    handle: "#ec4899",
  },
  blue: {
    border: "border-blue-500",
    bg: "bg-blue-900/80",
    text: "text-blue-300",
    handle: "#60a5fa",
  },
  red: {
    border: "border-red-500",
    bg: "bg-red-900",
    text: "text-red-200",
    handle: "#ef4444",
  },
  violet: {
    border: "border-violet-500",
    bg: "bg-violet-900",
    text: "text-violet-200",
    handle: "#8b5cf6",
  },
  cyan: {
    border: "border-cyan-500",
    bg: "bg-cyan-900",
    text: "text-cyan-200",
    handle: "#06b6d4",
  },
  orange: {
    border: "border-orange-500",
    bg: "bg-orange-900/80",
    text: "text-orange-300",
    handle: "#f97316",
  },
  green: {
    border: "border-green-500",
    bg: "bg-green-900",
    text: "text-green-200",
    handle: "#22c55e",
  },
  purple: {
    border: "border-purple-500",
    bg: "bg-purple-900",
    text: "text-purple-200",
    handle: "#a855f7",
  },
  indigo: {
    border: "border-indigo-500",
    bg: "bg-indigo-900",
    text: "text-indigo-200",
    handle: "#6366f1",
  },
  teal: {
    border: "border-teal-500",
    bg: "bg-teal-900",
    text: "text-teal-200",
    handle: "#14b8a6",
  },
  gray: {
    border: "border-gray-500",
    bg: "bg-gray-800",
    text: "text-gray-200",
    handle: "#6b7280",
  },
};

// ── Category → colour fallback (used when ui_config.color is empty) ──

export const CATEGORY_STYLES: Record<string, ColorSet> = {
  environment: COLOR_MAP.emerald,
  llm: COLOR_MAP.orange,
  prompt: COLOR_MAP.amber,
  policy: COLOR_MAP.blue,
  perception: COLOR_MAP.teal,
  processing: COLOR_MAP.indigo,
  decision: COLOR_MAP.pink,
  tool: COLOR_MAP.cyan,
  skill: COLOR_MAP.violet,
  agent: COLOR_MAP.purple,
  control: COLOR_MAP.red,
  boundary: COLOR_MAP.gray,
  server: COLOR_MAP.teal,
  output: COLOR_MAP.orange,
  example: COLOR_MAP.yellow,
  custom: COLOR_MAP.gray,
};

export const DEFAULT_STYLE: ColorSet = COLOR_MAP.gray;

// ── Handle positioning ──

export function handleTopPct(index: number, total: number): number {
  if (total === 1) return 50;
  return 15 + (index * 70) / Math.max(total - 1, 1);
}

// ── Port label helper ──

export function portLabel(
  port: PortSchema,
  data: Record<string, unknown>,
): string {
  if (port.name === "value" && data.portName) {
    return data.portName as string;
  }
  return port.name;
}
