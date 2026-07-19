/** Routed data edge — the wire for a data flow between two node handles.
 *
 * Two rendering modes, toggled globally via `useFlowStore.routingMode`:
 *  - "curved":     a plain bezier (identical to React Flow's default wire).
 *  - "orthogonal": a rounded polyline through the layout's `data.waypoints`
 *                  (the clear channels auto-layout reserves for long edges),
 *                  so the wire never disappears behind the nodes it crosses.
 *
 * When there are no waypoints (short edge, or graph never auto-laid-out) both
 * modes fall back to the bezier, so nothing ever renders as a broken line.
 */

import { memo } from "react";
import { BaseEdge, getBezierPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import { useFlowStore } from "../useFlowStore";

type Pt = { x: number; y: number };

/** Rounded polyline through `pts`, cutting each interior corner by `r`. */
function roundedPath(pts: Pt[], r = 12): string {
  if (pts.length < 2) return "";
  let d = `M ${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const p0 = pts[i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const d1 = Math.hypot(p1.x - p0.x, p1.y - p0.y) || 1;
    const d2 = Math.hypot(p2.x - p1.x, p2.y - p1.y) || 1;
    const rr = Math.min(r, d1 / 2, d2 / 2);
    const a = { x: p1.x - ((p1.x - p0.x) / d1) * rr, y: p1.y - ((p1.y - p0.y) / d1) * rr };
    const b = { x: p1.x + ((p2.x - p1.x) / d2) * rr, y: p1.y + ((p2.y - p1.y) / d2) * rr };
    d += ` L ${a.x},${a.y} Q ${p1.x},${p1.y} ${b.x},${b.y}`;
  }
  const last = pts[pts.length - 1];
  d += ` L ${last.x},${last.y}`;
  return d;
}

function RoutedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
  style = {},
}: EdgeProps) {
  const routingMode = useFlowStore((s) => s.routingMode);
  const waypoints = (data?.waypoints as Pt[] | undefined) ?? [];

  let path: string;
  if (routingMode === "orthogonal" && waypoints.length > 0) {
    path = roundedPath([
      { x: sourceX, y: sourceY },
      ...waypoints,
      { x: targetX, y: targetY },
    ]);
  } else {
    [path] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
  }

  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

export default memo(RoutedEdge);
