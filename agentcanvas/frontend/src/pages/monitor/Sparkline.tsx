/**
 * Minimal SVG sparkline. Values are 0–100 (percentages).
 * No chart library — keeps the bundle small and the visuals readable
 * at the densities we actually need (60-sample rolling buffer).
 */
import { useMemo } from "react";

export interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
  fill?: string;
  max?: number;
}

export default function Sparkline({
  values,
  width = 480,
  height = 80,
  stroke = "#60a5fa",
  fill = "rgba(96, 165, 250, 0.15)",
  max = 100,
}: SparklineProps) {
  const { path, area, last } = useMemo(() => {
    if (values.length === 0) return { path: "", area: "", last: 0 };
    const n = values.length;
    const stepX = n > 1 ? width / (n - 1) : width;
    const points = values.map((v, i) => {
      const x = i * stepX;
      const y = height - (Math.min(Math.max(v, 0), max) / max) * height;
      return [x, y] as const;
    });
    const path = points
      .map(
        ([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`,
      )
      .join(" ");
    const area =
      `M0,${height} ` +
      points.map(([x, y]) => `L${x.toFixed(1)},${y.toFixed(1)}`).join(" ") +
      ` L${width},${height} Z`;
    return { path, area, last: values[values.length - 1] };
  }, [values, width, height, max]);

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="block"
    >
      {/* horizontal grid at 25/50/75% */}
      {[0.25, 0.5, 0.75].map((f) => (
        <line
          key={f}
          x1={0}
          y1={height * (1 - f)}
          x2={width}
          y2={height * (1 - f)}
          stroke="#374151"
          strokeWidth={1}
          strokeDasharray="2 4"
        />
      ))}
      {area && <path d={area} fill={fill} />}
      {path && <path d={path} fill="none" stroke={stroke} strokeWidth={1.5} />}
      {values.length > 0 && (
        <circle
          cx={width}
          cy={height - (Math.min(Math.max(last, 0), max) / max) * height}
          r={2.5}
          fill={stroke}
        />
      )}
    </svg>
  );
}
