/** PointCloudLayout — 3-D orbit viewer for a fused point cloud (three.js).
 *
 * Renders the `pointCloudViewer` node from the live `viewer_data` fields
 * `{positions_b64, colors_b64, count, bbox, camera_centres}`. Positions/colors
 * arrive as base64 typed-array blocks (float32 XYZ / uint8 RGB) — decoded here
 * into a BufferGeometry. Camera auto-frames the bbox; OrbitControls let the user
 * orbit inside the node card. `nodrag/nopan/nowheel` isolate pointer + wheel
 * events from the React-Flow canvas so orbiting doesn't pan the graph.
 *
 * This module pulls in three.js / @react-three; GenericBlockRenderer lazy-loads
 * it so the WebGL bundle only ships when a pointCloudViewer is on the canvas.
 */

import { Handle, Position } from "@xyflow/react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
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

const W = 320; // default box size (px); user can resize from the corner
const H = 240;
const MIN_W = 180;
const MIN_H = 140;

interface PointCloudLayoutProps {
  id: string;
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}

interface CloudFields {
  positions_b64?: string;
  colors_b64?: string | null;
  count?: number;
  bbox?: [number[], number[]] | null;
  camera_centres?: number[][];
}

/** base64 (native little-endian, as written by numpy .tobytes()) → typed array.
 *  atob yields a fresh, 0-offset ArrayBuffer, so the Float32 view is aligned. */
function b64ToF32(b64: string | undefined | null): Float32Array | null {
  if (!b64) return null;
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
function b64ToU8(b64: string | undefined | null): Uint8Array | null {
  if (!b64) return null;
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function PointsScene({
  positions,
  colors,
  centres,
  pointSize,
}: {
  positions: Float32Array;
  colors: Uint8Array | null;
  centres: number[][] | undefined;
  pointSize: number;
}) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (colors && colors.length >= positions.length) {
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3, true)); // normalized uint8 → [0,1]
    }
    return g;
  }, [positions, colors]);
  useEffect(() => () => geom.dispose(), [geom]);

  const camGeom = useMemo(() => {
    if (!centres || centres.length === 0) return null;
    const arr = new Float32Array(centres.length * 3);
    centres.forEach((c, i) => {
      arr[i * 3] = c[0];
      arr[i * 3 + 1] = c[1];
      arr[i * 3 + 2] = c[2];
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [centres]);
  useEffect(() => () => camGeom?.dispose(), [camGeom]);

  const hasColors = !!colors && colors.length >= positions.length;
  return (
    <>
      <points geometry={geom}>
        <pointsMaterial
          size={pointSize}
          sizeAttenuation={false}
          vertexColors={hasColors}
          color={hasColors ? "#ffffff" : "#60a5fa"}
        />
      </points>
      {camGeom && (
        <points geometry={camGeom}>
          <pointsMaterial size={7} sizeAttenuation={false} color="#f97316" />
        </points>
      )}
    </>
  );
}

export default function PointCloudLayout({ id, data, schema }: PointCloudLayoutProps) {
  const nodeOutput = useNodeOutput(id) as
    | { fields?: CloudFields; status?: string }
    | undefined;
  const uiConfig = schema?.ui_config;
  const fields: CloudFields = nodeOutput?.fields || {};

  const positions = useMemo(() => b64ToF32(fields.positions_b64), [fields.positions_b64]);
  const colors = useMemo(() => b64ToU8(fields.colors_b64), [fields.colors_b64]);
  const count = fields.count || 0;
  const bbox = fields.bbox || null;
  const pointSize = Math.max(0.5, Number(data.point_size) || 1.5);

  // Camera framing from the bbox (recompute → remount Canvas via key).
  const cam = useMemo(() => {
    if (!bbox) return null;
    const [mn, mx] = bbox;
    const center: [number, number, number] = [
      (mn[0] + mx[0]) / 2,
      (mn[1] + mx[1]) / 2,
      (mn[2] + mx[2]) / 2,
    ];
    const dx = mx[0] - mn[0];
    const dy = mx[1] - mn[1];
    const dz = mx[2] - mn[2];
    const r = Math.max(0.1, Math.sqrt(dx * dx + dy * dy + dz * dz) / 2);
    return {
      center,
      position: [center[0] + r * 1.6, center[1] + r * 1.2, center[2] + r * 1.6] as [
        number,
        number,
        number,
      ],
      near: Math.max(r / 100, 0.001),
      far: r * 20 + 10,
    };
  }, [bbox]);

  const ports: PortSchema[] = schema?.input_ports || [];
  const category = schema?.category || "output";
  const normalizedCat = category.startsWith("server:") ? "server" : category;
  const style =
    (uiConfig?.color && COLOR_MAP[uiConfig.color]) ||
    CATEGORY_STYLES[normalizedCat] ||
    DEFAULT_STYLE;
  const label = (data.label as string) || schema?.display_name || "Point Cloud";
  const status = nodeOutput?.status || "idle";
  const dotClass = STATUS_DOT[status] || STATUS_DOT.idle;
  const ready = positions && positions.length > 0 && cam;

  // User-resizable viewport. `resize: both` gives a native corner grabber; a
  // ResizeObserver mirrors the dragged size back into state so a re-render
  // (new cloud) doesn't snap the box back to its default, and R3F's own
  // ResizeObserver re-fits the canvas — no letterbox gap at any size.
  const boxRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: W, h: H });
  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setDims((d) =>
        Math.abs(d.w - cr.width) > 1 || Math.abs(d.h - cr.height) > 1
          ? { w: Math.round(cr.width), h: Math.round(cr.height) }
          : d,
      );
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      className={`relative rounded-lg border-2 ${style.border} bg-gray-900 shadow-lg`}
      style={{ minWidth: MIN_W + 16 }}
    >
      {/* Header */}
      <div className={`flex items-center gap-2 rounded-t-lg px-3 py-1.5 ${style.bg}`}>
        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
        <span className={`flex-1 text-xs font-semibold ${style.text}`}>{label}</span>
        <span className="text-[9px] text-gray-400">
          {count ? `${count.toLocaleString()} pts` : "—"}
        </span>
      </div>

      {/* Body: three.js canvas. The nodrag/nopan/nowheel classes tell React
          Flow to leave pointer + wheel events alone so OrbitControls can orbit
          / zoom (same pattern as ImageGridViewerLayout). Do NOT stopPropagation
          in the capture phase — that would kill the event before it reaches the
          R3F canvas and the controls would never see a pointerdown. */}
      <div className="nopan nodrag nowheel p-2">
        <div
          ref={boxRef}
          className="ac-pc-box relative overflow-hidden rounded bg-gray-950"
          style={{
            width: dims.w,
            height: dims.h,
            minWidth: MIN_W,
            minHeight: MIN_H,
            resize: "both",
          }}
        >
          {/* The Canvas is sized in EXPLICIT PIXELS from the same `dims` that
              size the box — never `100%`. R3F's react-use-measure reads those
              exact pixels, so the drawing buffer always matches the box; a
              percentage height failed to resolve here and left the canvas at
              its 300x150 <canvas> default (the L-gap). Resizing the box updates
              `dims` (ResizeObserver) → the canvas follows in lockstep. */}
          {ready ? (
            <Canvas
              key={fields.positions_b64?.slice(0, 24)}
              style={{ width: "100%", height: "100%", display: "block" }}
              dpr={[1, 1.5]}
              gl={{ antialias: true, powerPreference: "low-power" }}
              camera={{ position: cam!.position, fov: 50, near: cam!.near, far: cam!.far }}
            >
              <color attach="background" args={["#0b0f19"]} />
              <PointsScene
                positions={positions!}
                colors={colors}
                centres={fields.camera_centres}
                pointSize={pointSize}
              />
              <OrbitControls makeDefault target={cam!.center} enableDamping={false} />
            </Canvas>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-[10px] text-gray-600">
              Waiting for cloud…
            </div>
          )}
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
