/** Renders tensor/ndarray/bytes __type markers — shape + dtype summary. */

import type { ValueRendererProps } from "./registry";

export default function TensorRenderer({ value }: ValueRendererProps) {
  if (value === null || value === undefined) {
    return <span className="text-gray-600 text-[10px]">—</span>;
  }

  if (typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    const typeName = String(obj.__type || "tensor");

    if (obj.__type === "bytes") {
      return (
        <span className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px]">
          <span className="text-amber-400">bytes</span>
          <span className="text-gray-400">{String(obj.length)} B</span>
        </span>
      );
    }

    return (
      <span className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px]">
        <span className="text-purple-400">{typeName}</span>
        {obj.shape != null && (
          <span className="text-gray-300">
            [{(obj.shape as number[]).join(" x ")}]
          </span>
        )}
        {obj.dtype != null && (
          <span className="text-gray-500">{String(obj.dtype)}</span>
        )}
      </span>
    );
  }

  return (
    <span className="text-gray-500 text-[10px]">
      {String(value).slice(0, 80)}
    </span>
  );
}
