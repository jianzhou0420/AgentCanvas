/** TrajectoryLayout — top-down camera-trajectory sink (SVG bird's-eye).
 *
 * Renders the `trajectoryViewer` node: the live `viewer_data` fields
 * `{est_path, gt_path, current, points, axes}` (2-D polylines already projected
 * backend-side) as an auto-fit bird's-eye plot — estimated path (blue) vs
 * ground truth (grey) plus a faint map-point scatter. Hand-rolled SVG, no chart
 * library (same choice as pages/monitor/Sparkline.tsx). Ports are class-level
 * (schema.input_ports), unlike the config-driven ImageGridViewerLayout.
 */

import { Handle, Position } from "@xyflow/react";
import { useMemo } from "react";
import { useNodeOutput } from "./useNodeOutput";
import {
  COLOR_MAP,
  CATEGORY_STYLES,
  DEFAULT_STYLE,
  getWireColor,
  handleTopPct,
} from "./layoutUtils";
import type { NodeSchema, PortSchema } from "./layoutUtils";

const STATUS_DOT: Record<string, string> = {
  idle: "bg-gray-500",
  running: "bg-green-400 animate-pulse",
  done: "bg-blue-400",
  error: "bg-red-400",
};

const W = 280;
const H = 200;
const PAD = 12;
const SCATTER_CAP = 1500; // strided render cap — keep the SVG light

interface TrajectoryLayoutProps {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}

type Pt = [number, number];

interface TrajFields {
  est_path?: Pt[];
  gt_path?: Pt[];
  current?: Pt | null;
  points?: Pt[];
  axes?: string;
}

/** Project data-space (u,v) into the SVG box, aspect-preserving, y-flipped. */
function makeProjector(pts: Pt[]) {
  if (pts.length === 0) {
    return { px: (u: number) => u, py: (v: number) => v, empty: true };
  }
  let minU = Infinity,
    maxU = -Infinity,
    minV = Infinity,
    maxV = -Infinity;
  for (const [u, v] of pts) {
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  const dw = maxU - minU || 1;
  const dh = maxV - minV || 1;
  const s = Math.min((W - 2 * PAD) / dw, (H - 2 * PAD) / dh);
  const ox = (W - s * dw) / 2;
  const oy = (H - s * dh) / 2;
  return {
    px: (u: number) => ox + (u - minU) * s,
    py: (v: number) => H - (oy + (v - minV) * s), // flip so +v is up
    empty: false,
  };
}

function polyPath(pts: Pt[], px: (u: number) => number, py: (v: number) => number): string {
  return pts
    .map(([u, v], i) => `${i === 0 ? "M" : "L"}${px(u).toFixed(1)},${py(v).toFixed(1)}`)
    .join(" ");
}

export default function TrajectoryLayout({ id, data, schema }: TrajectoryLayoutProps) {
  const nodeOutput = useNodeOutput(id) as
    | { fields?: TrajFields; status?: string }
    | undefined;
  const uiConfig = schema?.ui_config;
  const fields: TrajFields = nodeOutput?.fields || {};

  const est = fields.est_path || [];
  const gt = fields.gt_path || [];
  const scatter = fields.points || [];
  const current = fields.current || null;
  const axes = fields.axes || (data.axes as string) || "XZ";

  const { estPath, gtPath, dots, cur, empty } = useMemo(() => {
    const all: Pt[] = [...est, ...gt, ...scatter];
    const proj = makeProjector(all);
    if (proj.empty) {
      return { estPath: "", gtPath: "", dots: [] as Pt[], cur: null as Pt | null, empty: true };
    }
    const stride = Math.max(1, Math.ceil(scatter.length / SCATTER_CAP));
    const dots: Pt[] = [];
    for (let i = 0; i < scatter.length; i += stride) {
      dots.push([proj.px(scatter[i][0]), proj.py(scatter[i][1])]);
    }
    return {
      estPath: polyPath(est, proj.px, proj.py),
      gtPath: polyPath(gt, proj.px, proj.py),
      dots,
      cur: current ? ([proj.px(current[0]), proj.py(current[1])] as Pt) : null,
      empty: false,
    };
  }, [est, gt, scatter, current]);

  const ports: PortSchema[] = schema?.input_ports || [];
  const category = schema?.category || "output";
  const normalizedCat = category.startsWith("server:") ? "server" : category;
  const style =
    (uiConfig?.color && COLOR_MAP[uiConfig.color]) ||
    CATEGORY_STYLES[normalizedCat] ||
    DEFAULT_STYLE;
  const label = (data.label as string) || schema?.display_name || "Trajectory";
  const status = nodeOutput?.status || "idle";
  const dotClass = STATUS_DOT[status] || STATUS_DOT.idle;

  return (
    <div
      className={`relative rounded-lg border-2 ${style.border} bg-gray-900 shadow-lg`}
      style={{ minWidth: W + 24 }}
    >
      {/* Header */}
      <div className={`flex items-center gap-2 rounded-t-lg px-3 py-1.5 ${style.bg}`}>
        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
        <span className={`flex-1 text-xs font-semibold ${style.text}`}>{label}</span>
        <span className="text-[9px] text-gray-400">
          {axes} · {est.length}f{scatter.length ? ` · ${scatter.length}pt` : ""}
        </span>
      </div>

      {/* Body: SVG bird's-eye */}
      <div className="nopan nodrag nowheel p-2">
        <svg
          width="100%"
          viewBox={`0 0 ${W} ${H}`}
          className="block rounded bg-gray-950"
          style={{ aspectRatio: `${W} / ${H}` }}
        >
          <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="none" stroke="#1f2937" />
          {empty ? (
            <text x={W / 2} y={H / 2} fill="#4b5563" fontSize="10" textAnchor="middle">
              Waiting for poses…
            </text>
          ) : (
            <>
              {/* map-point scatter */}
              {dots.map(([x, y], k) => (
                <circle key={k} cx={x.toFixed(1)} cy={y.toFixed(1)} r={0.8} fill="#475569" />
              ))}
              {/* ground-truth path */}
              {gtPath && (
                <path d={gtPath} fill="none" stroke="#9ca3af" strokeWidth={1.2} strokeDasharray="3 3" />
              )}
              {/* estimated path */}
              {estPath && <path d={estPath} fill="none" stroke="#60a5fa" strokeWidth={1.6} />}
              {/* current pose */}
              {cur && <circle cx={cur[0].toFixed(1)} cy={cur[1].toFixed(1)} r={3} fill="#f97316" />}
            </>
          )}
        </svg>
        {/* legend */}
        <div className="mt-1 flex items-center justify-center gap-3 text-[9px] text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block h-0.5 w-3" style={{ background: "#60a5fa" }} /> est
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-0.5 w-3" style={{ background: "#9ca3af" }} /> gt
          </span>
        </div>
      </div>

      {/* Input handles (left) — one per class-level port */}
      {ports.map((port, i) => (
        <Handle
          key={`in-${port.name}`}
          type="target"
          position={Position.Left}
          id={port.name}
          style={{
            top: `${handleTopPct(i, ports.length)}%`,
            background: getWireColor(port.wire_type),
            width: 10,
            height: 10,
            border: "2px solid #1f2937",
          }}
        />
      ))}

      {/* State handle (bottom center) */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="__state__"
        className="state-handle"
        style={{ background: "#8b5cf6", width: 6, height: 6, border: "1.5px solid #1f2937" }}
      />
    </div>
  );
}
