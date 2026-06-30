/** Renders primitive values with type-colored display. */

import type { ValueRendererProps } from "./registry";

export default function PrimitiveRenderer({ value }: ValueRendererProps) {
  if (value === null || value === undefined) {
    return <span className="text-gray-500 text-[10px] italic">null</span>;
  }

  if (typeof value === "boolean") {
    return (
      <span className="text-purple-400 text-[11px] font-mono">
        {value ? "true" : "false"}
      </span>
    );
  }

  if (typeof value === "number") {
    return (
      <span className="text-blue-400 text-[11px] font-mono">
        {Number.isInteger(value) ? value : value.toFixed(4)}
      </span>
    );
  }

  // String
  const s = String(value);
  if (s.length <= 80 && !s.includes("\n")) {
    return <span className="text-green-400 text-[11px]">{s}</span>;
  }

  return (
    <pre className="text-[10px] text-gray-300 bg-gray-800 p-1.5 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
      {s}
    </pre>
  );
}
