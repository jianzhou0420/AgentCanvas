/** Block layout — standard rectangle node with title, config fields, and ports. */

import { Handle, Position, useReactFlow } from "@xyflow/react";
import { useCallback, useState } from "react";
import {
  COLOR_MAP,
  CATEGORY_STYLES,
  DEFAULT_STYLE,
  getWireColor,
  handleTopPct,
  isListWireType,
  portLabel,
} from "./layoutUtils";
import type { NodeSchema, UIConfigSchema } from "./layoutUtils";
import { ConfigFieldRenderer } from "./ConfigFieldRenderer";

export default function BlockLayout({
  id,
  data,
  schema,
  uiConfig,
}: {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
  uiConfig: UIConfigSchema | undefined;
}) {
  const label =
    (data.label as string) ||
    (data.toolName as string) ||
    schema?.display_name ||
    "Block";
  const category = schema?.category || "custom";
  const description =
    (data.toolDescription as string) || schema?.description || "";
  const inputPorts = schema?.input_ports || [];
  const outputPorts = schema?.output_ports || [];
  const configFields = uiConfig?.config_fields || [];

  const normalizedCat = category.startsWith("server:") ? "server" : category;
  const style =
    (uiConfig?.color && COLOR_MAP[uiConfig.color]) ||
    CATEGORY_STYLES[normalizedCat] ||
    DEFAULT_STYLE;

  // Collapsible: nodes with >=4 config fields or >=6 input ports can fold
  const canCollapse = configFields.length >= 4 || inputPorts.length >= 6;
  const [collapsed, setCollapsed] = useState(
    data._collapsed !== undefined ? Boolean(data._collapsed) : canCollapse,
  );

  const maxPorts = Math.max(inputPorts.length, outputPorts.length, 1);
  const configHeight = collapsed ? 0 : configFields.length * 18;
  const portHeight = collapsed ? Math.min(maxPorts, 3) * 16 : maxPorts * 22;
  const height = collapsed
    ? Math.max(36, 20 + portHeight)
    : Math.max(60, 28 + maxPorts * 22 + configHeight);

  const { setNodes } = useReactFlow();
  const toggleCollapse = useCallback(() => {
    const next = !collapsed;
    setCollapsed(next);
    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, _collapsed: next } } : n,
      ),
    );
  }, [collapsed, id, setNodes]);

  const minWidth = uiConfig?.min_width ? parseInt(uiConfig.min_width) : 120;
  const maxWidth = uiConfig?.max_width
    ? parseInt(uiConfig.max_width)
    : undefined;

  return (
    <div
      className={`relative rounded border-2 ${style.border} ${style.bg} px-3 py-2 shadow-md`}
      style={{ minWidth, maxWidth, minHeight: height }}
    >
      {/* Title bar */}
      <div className="flex items-center justify-center gap-1">
        {canCollapse && (
          <button
            className="nopan nodrag text-[9px] text-gray-500 hover:text-gray-300"
            onClick={toggleCollapse}
            title={collapsed ? "Expand" : "Collapse"}
          >
            {collapsed ? "▶" : "▼"}
          </button>
        )}
        <div className={`text-center text-xs font-semibold ${style.text}`}>
          {label}
        </div>
      </div>

      {/* Collapsible body — overflow hidden preserves width when folded */}
      <div
        style={{
          overflow: "hidden",
          maxHeight: collapsed ? 0 : 9999,
          transition: "max-height 0.15s ease",
        }}
      >
        {/* Description (truncated) */}
        {description && (
          <div
            className="text-center text-[9px] text-gray-500"
            title={description}
          >
            {description.length > 30
              ? description.slice(0, 28) + "…"
              : description}
          </div>
        )}

        {/* Config fields */}
        {configFields.length > 0 && (
          <div className="mt-1 space-y-0.5">
            {configFields.map((f) => (
              <ConfigFieldRenderer
                key={f.name}
                field={f}
                data={data}
                nodeId={id}
              />
            ))}
          </div>
        )}
      </div>

      {/* Input handles (left) — always rendered for edge connectivity */}
      {inputPorts.map((port, i) => {
        const topPct = handleTopPct(i, inputPorts.length);
        const color =
          (uiConfig?.color && style.handle) || getWireColor(port.wire_type);
        const isList = isListWireType(port.wire_type);
        const displayName = portLabel(port, data);
        return (
          <div key={`in-${port.name}`}>
            <Handle
              type="target"
              position={Position.Left}
              id={port.name}
              style={{
                top: `${topPct}%`,
                background: color,
                width: collapsed ? 7 : 10,
                height: collapsed ? 7 : 10,
                border: isList ? `2px double ${color}` : "2px solid #1f2937",
                boxShadow: isList ? `0 0 0 1px #1f2937` : undefined,
              }}
            />
            {!collapsed && (
              <div
                className="absolute text-[7px] text-gray-400"
                style={{
                  left: -2,
                  top: `${topPct - 4}%`,
                  transform: "translateX(-100%)",
                }}
              >
                {displayName}
              </div>
            )}
          </div>
        );
      })}

      {/* Output handles (right) — always rendered for edge connectivity */}
      {outputPorts.map((port, i) => {
        const topPct = handleTopPct(i, outputPorts.length);
        const color =
          (uiConfig?.color && style.handle) || getWireColor(port.wire_type);
        const isList = isListWireType(port.wire_type);
        const displayName = portLabel(port, data);
        return (
          <div key={`out-${port.name}`}>
            <Handle
              type="source"
              position={Position.Right}
              id={port.name}
              style={{
                top: `${topPct}%`,
                background: color,
                width: collapsed ? 7 : 10,
                height: collapsed ? 7 : 10,
                border: isList ? `2px double ${color}` : "2px solid #1f2937",
                boxShadow: isList ? `0 0 0 1px #1f2937` : undefined,
              }}
            />
            {!collapsed && (
              <div
                className="absolute text-[7px] text-gray-400"
                style={{
                  right: -2,
                  top: `${topPct - 4}%`,
                  transform: "translateX(100%)",
                }}
              >
                {displayName}
              </div>
            )}
          </div>
        );
      })}

      {/* State handle (bottom center) — toggled via .show-state-edges CSS class on container */}
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
