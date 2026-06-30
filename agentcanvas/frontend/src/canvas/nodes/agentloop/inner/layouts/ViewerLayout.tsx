/** Viewer layout — output viewer node driven by display_fields in ui_config.
 *
 * Each viewer node receives per-instance data via viewer_data WS events
 * (keyed by node_id). The display type registry dispatches to the correct
 * field renderer component.
 *
 * Accumulate fields (log_list with accumulate=true) are handled by the
 * backend — _SinkBase builds the full array in persistent node state and
 * sends it each time. The frontend simply renders what it receives.
 */

import { Suspense } from "react";
import { Handle, Position } from "@xyflow/react";
import { useNodeOutput } from "./useNodeOutput";
import { resolveFieldRenderer } from "./viewerFieldRegistry";
import {
  COLOR_MAP,
  CATEGORY_STYLES,
  DEFAULT_STYLE,
  getWireColor,
  isListWireType,
  handleTopPct,
} from "./layoutUtils";
import type { NodeSchema, DisplayFieldSchema } from "./layoutUtils";

const STATUS_DOT: Record<string, string> = {
  idle: "bg-gray-500",
  running: "bg-green-400 animate-pulse",
  done: "bg-blue-400",
  error: "bg-red-400",
};

interface ViewerLayoutProps {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}

/** Resolve the display value for a field from nodeOutput.
 *  New path: fields bag from viewer_data WS events.
 *  Legacy fallback: direct key on nodeOutput (backward compat). */
function getFieldValue(
  field: DisplayFieldSchema,
  nodeOutput: unknown,
): unknown {
  if (!nodeOutput || typeof nodeOutput !== "object") return undefined;
  const out = nodeOutput as Record<string, unknown>;
  // New path: fields bag
  const fields = out.fields as Record<string, unknown> | undefined;
  if (fields && fields[field.data_key] !== undefined) {
    return fields[field.data_key];
  }
  // Legacy fallback: direct key
  return out[field.data_key];
}

export default function ViewerLayout({ id, data, schema }: ViewerLayoutProps) {
  const nodeOutput = useNodeOutput(id);
  const uiConfig = schema?.ui_config;
  const displayFields = uiConfig?.display_fields || [];
  const inputPorts = schema?.input_ports || [];

  const category = schema?.category || "output";
  const normalizedCat = category.startsWith("server:") ? "server" : category;
  const style =
    (uiConfig?.color && COLOR_MAP[uiConfig.color]) ||
    CATEGORY_STYLES[normalizedCat] ||
    DEFAULT_STYLE;

  const label = (data.label as string) || schema?.display_name || "Viewer";
  const status: string =
    ((nodeOutput as unknown as Record<string, unknown> | undefined)
      ?.status as string) || "idle";
  const dotClass = STATUS_DOT[status] || STATUS_DOT.idle;

  const minWidth = uiConfig?.min_width ? parseInt(uiConfig.min_width) : 220;
  const maxWidth = uiConfig?.max_width ? parseInt(uiConfig.max_width) : 400;

  return (
    <div
      className={`relative rounded-lg border-2 ${style.border} bg-gray-900 shadow-lg`}
      style={{ minWidth, maxWidth }}
    >
      {/* Header */}
      <div
        className={`flex items-center gap-2 rounded-t-lg px-3 py-1.5 ${style.bg}`}
      >
        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
        <span className={`flex-1 text-xs font-semibold ${style.text}`}>
          {label}
        </span>
      </div>

      {/* Body */}
      <div className="nopan nodrag nowheel flex flex-col gap-2 p-3 text-xs text-gray-300">
        {displayFields.length === 0 ? (
          <div className="italic text-gray-600">
            No display fields configured
          </div>
        ) : (
          displayFields.map((field, i) => {
            const Renderer = resolveFieldRenderer(field.display_type);
            if (!Renderer) return null;

            const value = getFieldValue(field, nodeOutput);

            return (
              <Suspense
                key={field.name ?? i}
                fallback={
                  <div className="text-[10px] text-gray-600">Loading...</div>
                }
              >
                <Renderer
                  value={value}
                  label={field.label}
                  max_visible={field.max_visible}
                />
              </Suspense>
            );
          })
        )}
      </div>

      {/* Input handles (left) */}
      {inputPorts.map((port, i) => {
        const topPct = handleTopPct(i, inputPorts.length);
        const color = getWireColor(port.wire_type);
        const isList = isListWireType(port.wire_type);
        return (
          <Handle
            key={`in-${port.name}`}
            type="target"
            position={Position.Left}
            id={port.name}
            style={{
              top: `${topPct}%`,
              background: color,
              width: 10,
              height: 10,
              border: isList ? `2px double ${color}` : "2px solid #1f2937",
              boxShadow: isList ? `0 0 0 1px #1f2937` : undefined,
            }}
          />
        );
      })}

      {/* State handle (bottom center) */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="__state__"
        className="state-handle"
        style={{
          background: "#8b5cf6",
          width: 6,
          height: 6,
          border: "1.5px solid #1f2937",
        }}
      />
    </div>
  );
}
