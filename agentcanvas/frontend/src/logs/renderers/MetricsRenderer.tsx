/** Renders METRICS wire type as mini card grid. */

import type { ValueRendererProps } from "./registry";

export default function MetricsRenderer({ value }: ValueRendererProps) {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return <span className="text-gray-600 text-[10px]">no metrics</span>;
    }
  }
  if (value === null || value === undefined || typeof value !== "object") {
    return <span className="text-gray-600 text-[10px]">no metrics</span>;
  }

  const entries = Object.entries(value as Record<string, unknown>).filter(
    ([, v]) => typeof v === "number",
  );

  if (entries.length === 0) {
    return <span className="text-gray-600 text-[10px]">empty metrics</span>;
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([key, val]) => (
        <div
          key={key}
          className="flex flex-col rounded border border-gray-700 bg-gray-800/50 px-2 py-1"
        >
          <span className="text-[9px] text-gray-500 uppercase">{key}</span>
          <span className="text-[12px] font-mono text-blue-300">
            {typeof val === "number"
              ? val < 1
                ? val.toFixed(4)
                : val.toFixed(2)
              : String(val)}
          </span>
        </div>
      ))}
    </div>
  );
}
