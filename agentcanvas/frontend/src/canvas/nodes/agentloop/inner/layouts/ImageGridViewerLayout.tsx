/** ImageGridViewerLayout — configurable multi-image/depth sink.
 *
 * Ports and grid shape are driven by per-instance config (ADR note: rows,
 * cols, ports[] stored at config root). Each port becomes a left-edge
 * input handle; each cell renders `nodeOutput.fields[port.name]` as a
 * base64 PNG. No class-level ports — the schema publishes `ports_mode:
 * "sink"` and `portResolution.ts` derives ports from `config.ports`.
 */

import { Handle, Position } from "@xyflow/react";
import { useNodeOutput } from "./useNodeOutput";
import {
  COLOR_MAP,
  CATEGORY_STYLES,
  DEFAULT_STYLE,
  getWireColor,
  handleTopPct,
} from "./layoutUtils";
import type { NodeSchema } from "./layoutUtils";

const STATUS_DOT: Record<string, string> = {
  idle: "bg-gray-500",
  running: "bg-green-400 animate-pulse",
  done: "bg-blue-400",
  error: "bg-red-400",
};

interface PortConfig {
  name: string;
  wire_type: string;
}

interface ImageGridViewerLayoutProps {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}

const DEFAULT_PORTS: PortConfig[] = [{ name: "rgb", wire_type: "IMAGE" }];

function clampDim(v: unknown, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n) || n < 1) return fallback;
  return Math.min(Math.max(Math.round(n), 1), 4);
}

export default function ImageGridViewerLayout({
  id,
  data,
  schema,
}: ImageGridViewerLayoutProps) {
  const nodeOutput = useNodeOutput(id);
  const uiConfig = schema?.ui_config;

  const rows = clampDim(data.rows, 1);
  const cols = clampDim(data.cols, 1);
  const rawPorts = Array.isArray(data.ports)
    ? (data.ports as PortConfig[])
    : DEFAULT_PORTS;
  const ports = rawPorts.filter((p) => p && typeof p.name === "string");

  const category = schema?.category || "output";
  const normalizedCat = category.startsWith("server:") ? "server" : category;
  const style =
    (uiConfig?.color && COLOR_MAP[uiConfig.color]) ||
    CATEGORY_STYLES[normalizedCat] ||
    DEFAULT_STYLE;

  const label =
    (data.label as string) || schema?.display_name || "Image Viewer";
  const status: string =
    ((nodeOutput as unknown as Record<string, unknown> | undefined)
      ?.status as string) || "idle";
  const dotClass = STATUS_DOT[status] || STATUS_DOT.idle;

  const fields =
    ((nodeOutput as unknown as Record<string, unknown> | undefined)?.fields as
      | Record<string, string | string[]>
      | undefined) || {};

  const cellMinWidth = 120;
  const bodyMinWidth = cols * cellMinWidth + 24;
  const minWidth = Math.max(220, bodyMinWidth);

  return (
    <div
      className={`relative rounded-lg border-2 ${style.border} bg-gray-900 shadow-lg`}
      style={{ minWidth }}
    >
      {/* Header */}
      <div
        className={`flex items-center gap-2 rounded-t-lg px-3 py-1.5 ${style.bg}`}
      >
        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
        <span className={`flex-1 text-xs font-semibold ${style.text}`}>
          {label}
        </span>
        <span className="text-[9px] text-gray-400">
          {rows}×{cols} · {ports.length} port{ports.length === 1 ? "" : "s"}
        </span>
      </div>

      {/* Body: grid of image cells */}
      <div
        className="nopan nodrag nowheel grid gap-1 p-2"
        style={{
          gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
          gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        }}
      >
        {ports.length === 0 ? (
          <div className="col-span-full text-center italic text-gray-600 text-[10px]">
            No ports configured
          </div>
        ) : (
          ports.map((port) => {
            // LIST[IMAGE]/LIST[DEPTH] ports arrive as an array of base64 tiles;
            // single IMAGE/DEPTH ports as one string.
            const raw = fields[port.name];
            const imgs = Array.isArray(raw)
              ? raw.filter(Boolean)
              : raw
                ? [raw]
                : [];
            return (
              <div
                key={port.name}
                className="overflow-hidden rounded border border-gray-700 bg-gray-800"
              >
                <div className="border-b border-gray-700 px-1.5 py-0.5 text-center text-[10px] text-gray-500">
                  {port.name}
                  {imgs.length > 1 ? ` · ${imgs.length}` : ""}
                </div>
                {imgs.length === 0 ? (
                  <div className="flex items-center justify-center py-4 text-[9px] text-gray-600">
                    Waiting…
                  </div>
                ) : imgs.length === 1 ? (
                  <img
                    src={`data:image/jpeg;base64,${imgs[0]}`}
                    alt={port.name}
                    className="w-full object-contain"
                  />
                ) : (
                  <div
                    className="grid gap-0.5 p-0.5"
                    style={{
                      gridTemplateColumns: `repeat(${Math.round(Math.sqrt(imgs.length))}, 1fr)`,
                    }}
                  >
                    {imgs.map((b64, k) => (
                      <img
                        key={k}
                        src={`data:image/jpeg;base64,${b64}`}
                        alt={`${port.name}[${k}]`}
                        className="w-full rounded-sm object-cover aspect-[4/3]"
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Input handles (left) — one per port, evenly spaced */}
      {ports.map((port, i) => {
        const topPct = handleTopPct(i, ports.length);
        const color = getWireColor(port.wire_type);
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
              border: "2px solid #1f2937",
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
