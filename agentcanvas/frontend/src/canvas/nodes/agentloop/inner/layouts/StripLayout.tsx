/** Strip layout — narrow vertical gate node (IterIn, IterOut, GraphIn, GraphOut, OutputPort). */

import { useCallback, useState } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import {
  COLOR_MAP,
  DEFAULT_STYLE,
  WIRE_COLORS,
  handleTopPct,
  portLabel,
} from "./layoutUtils";
import type { NodeSchema, UIConfigSchema } from "./layoutUtils";
import { ConfigFieldRenderer } from "./ConfigFieldRenderer";
import { IterInExpandPanel } from "./IterInExpandPanel";

export default function StripLayout({
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
  const label = (data.label as string) || schema?.display_name || "Gate";
  const inputPorts = schema?.input_ports || [];
  const outputPorts = schema?.output_ports || [];
  const configFields = uiConfig?.config_fields || [];
  const nodeType = schema?.type as string | undefined;
  const isIterIn = nodeType === "iterIn";
  // Two-sided iterIn: its init-origin output ports double as LEFT input
  // handles — the run-start seed wires into the left, the body reads the
  // matching right-side output. An init port is one pass-through slot shown
  // on both sides; iterout-origin ports stay output-only.
  const leftPorts = isIterIn
    ? outputPorts.filter((p) => (p as { origin?: string }).origin === "init")
    : inputPorts;
  // iterIn has no user-editable config — but its expand panel still aggregates
  // persist toggles for the writer ports, so the chevron is shown whenever the
  // node has any output ports.
  const canExpand =
    configFields.length > 0 || (isIterIn && outputPorts.length > 0);

  const style = (uiConfig?.color && COLOR_MAP[uiConfig.color]) || DEFAULT_STYLE;
  const width = uiConfig?.width || "40px";
  const minHeight = uiConfig?.min_height || "80px";
  const rounding = uiConfig?.rounding || "rounded";

  const initialExpanded = Boolean(data._expanded);
  const [expanded, setExpanded] = useState(initialExpanded);
  const { setNodes } = useReactFlow();
  const toggleExpand = useCallback(() => {
    const next = !expanded;
    setExpanded(next);
    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, _expanded: next } } : n,
      ),
    );
  }, [expanded, id, setNodes]);

  const renderOutputRow = (
    port: { name: string; wire_type: string },
    topPct: number,
  ) => {
    const color = style.handle;
    const displayName = portLabel(port as never, data);
    return (
      <div key={`out-${port.name}`}>
        <Handle
          type="source"
          position={Position.Right}
          id={port.name}
          style={{
            top: `${topPct}%`,
            background: color,
            width: 8,
            height: 8,
            border: "1.5px solid #1f2937",
          }}
        />
        <div
          className="absolute text-[6px] text-gray-400"
          style={{
            right: -2,
            top: `${topPct - 3}%`,
            transform: "translateX(100%)",
          }}
        >
          {displayName}
        </div>
      </div>
    );
  };

  return (
    <div className="relative flex items-start">
      <div
        className={`relative border-2 ${style.border} ${style.bg} py-1 shadow-md ${rounding}`}
        style={{ width, minHeight }}
      >
        {/* Vertical label */}
        <div
          className={`absolute inset-0 flex items-center justify-center text-[7px] font-bold tracking-wider ${style.text}`}
          style={{ writingMode: "vertical-rl", textOrientation: "mixed" }}
        >
          {label.toUpperCase()}
        </div>

        {/* Expand / collapse chevron (only when config_fields exist) */}
        {canExpand && (
          <button
            className="nopan nodrag absolute -top-2 left-1/2 -translate-x-1/2 rounded-full border border-gray-700 bg-gray-900 px-1 text-[8px] leading-none text-gray-400 hover:text-gray-200"
            onClick={toggleExpand}
            title={expanded ? "Collapse" : "Expand config"}
          >
            {expanded ? "◀" : "▶"}
          </button>
        )}

        {/* Input handles (left) — for iterIn these are the init-origin ports
          (two-sided model); for other strip nodes, the schema input ports. */}
        {leftPorts.map((port, i) => {
          const topPct = handleTopPct(i, leftPorts.length);
          const color =
            port.wire_type === "BOOL" ? WIRE_COLORS.BOOL : style.handle;
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
                  width: 8,
                  height: 8,
                  border: "1.5px solid #1f2937",
                }}
              />
              <div
                className="absolute text-[6px] text-gray-400"
                style={{
                  left: -2,
                  top: `${topPct - 3}%`,
                  transform: "translateX(-100%)",
                }}
              >
                {displayName}
              </div>
            </div>
          );
        })}

        {/* Output handles (right) — iterIn renders in two bands with a divider;
          other strip nodes use a flat list. */}
        {isIterIn
          ? (() => {
              const initPorts = outputPorts.filter((p) => p.origin === "init");
              const iterOutPorts = outputPorts.filter(
                (p) => p.origin === "iterOut",
              );

              // Always-prefix synthesis → two disjoint bands. Init 8-42%, iterOut 58-92%.
              const initRows = initPorts.map((p, i) => {
                const span = 34;
                const top =
                  8 +
                  (initPorts.length === 1
                    ? span / 2
                    : (i * span) / Math.max(initPorts.length - 1, 1));
                return renderOutputRow(p, top);
              });
              const iterRows = iterOutPorts.map((p, i) => {
                const span = 34;
                const top =
                  58 +
                  (iterOutPorts.length === 1
                    ? span / 2
                    : (i * span) / Math.max(iterOutPorts.length - 1, 1));
                return renderOutputRow(p, top);
              });
              return (
                <>
                  {initRows}
                  {iterRows}
                  {/* Divider at 50% */}
                  <div
                    className="pointer-events-none absolute left-1 right-1"
                    style={{
                      top: "50%",
                      height: 0,
                      borderTop: `1px dashed ${style.handle}`,
                      opacity: 0.6,
                    }}
                  />
                </>
              );
            })()
          : outputPorts.map((port, i) =>
              renderOutputRow(port, handleTopPct(i, outputPorts.length)),
            )}

        {/* State handle (bottom center) — visible only when state toggle is on */}
        <Handle
          type="source"
          position={Position.Bottom}
          id="__state__"
          className="state-handle"
          style={{
            background: "#8b5cf6",
            width: 5,
            height: 5,
            border: "1px solid #1f2937",
          }}
        />
      </div>

      {/* Expanded side panel — config editor rendered next to the strip */}
      {expanded && canExpand && (
        <div
          className={`ml-1 rounded border ${style.border} bg-gray-900/95 px-2 py-1 shadow-md`}
          style={{ width: 240 }}
        >
          <div className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-gray-500">
            {label}
          </div>
          {isIterIn ? (
            <IterInExpandPanel
              iterInId={id}
              ports={
                outputPorts as unknown as Array<{
                  name: string;
                  wire_type: string;
                  persist?: boolean;
                  origin?: "init" | "iterOut";
                  writer_name?: string;
                }>
              }
            />
          ) : (
            <div className="space-y-1">
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
      )}
    </div>
  );
}
