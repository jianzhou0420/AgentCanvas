/** Metric table display field for viewer nodes.
 * Renders key-value pairs for metrics (SPL, SR, nDTW, etc.). */

import type { DisplayFieldProps } from "../viewerFieldRegistry";

export default function MetricTableField({ value, label }: DisplayFieldProps) {
  const metrics: Record<string, unknown> | null =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;

  return (
    <div>
      {label && <div className="mb-0.5 text-[9px] text-gray-500">{label}</div>}
      {!metrics ? (
        <div className="italic text-gray-600 text-[10px]">
          Available after episode completes
        </div>
      ) : (
        <div className="space-y-0.5">
          {Object.entries(metrics).map(([k, v]) => (
            <div key={k} className="flex justify-between text-[10px]">
              <span className="text-gray-400">{k}</span>
              <span className="font-mono text-gray-200">
                {typeof v === "number" ? v.toFixed(3) : String(v)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
