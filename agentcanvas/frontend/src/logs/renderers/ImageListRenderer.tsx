/** Renders LIST[IMAGE] / LIST[DEPTH] values as a tiled thumbnail grid.
 *
 * New logs carry `{ __type: "image_list", wire_type, count, items: [...] }`
 * where each item is an asset ref (`{ __type: "asset", path }`) or a base64
 * string. Older logs (pre image_list support) only carry `{ count }` — we show
 * a compact count badge for those so nothing breaks.
 */

import { useContext, useState } from "react";
import type { ValueRendererProps } from "./registry";
import { LogContext } from "../LogContext";
import { logApi } from "../logApi";

function resolveSrc(item: unknown, executionId: string | null): string | null {
  if (typeof item === "string") {
    if (item.length < 100) return null;
    return item.startsWith("data:") ||
      item.startsWith("http") ||
      item.startsWith("/")
      ? item
      : `data:image/jpeg;base64,${item}`;
  }
  if (item && typeof item === "object") {
    const obj = item as Record<string, unknown>;
    if (obj.__type === "asset" && typeof obj.path === "string" && executionId) {
      return logApi.assetUrl(executionId, obj.path);
    }
  }
  return null;
}

function Tile({ src, alt }: { src: string; alt: string }) {
  const [zoomed, setZoomed] = useState(false);
  return (
    <>
      <img
        src={src}
        alt={alt}
        loading="lazy"
        className="h-20 w-20 cursor-pointer rounded border border-gray-700 object-cover transition-transform hover:scale-105"
        onClick={() => setZoomed(true)}
      />
      {zoomed && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setZoomed(false)}
        >
          <img
            src={src}
            alt={alt}
            className="max-h-[90vh] max-w-[90vw] rounded"
          />
        </div>
      )}
    </>
  );
}

export default function ImageListRenderer({
  value,
  label,
}: ValueRendererProps) {
  const { executionId } = useContext(LogContext);

  const obj =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;

  const items =
    obj && Array.isArray(obj.items) ? (obj.items as unknown[]) : null;
  const wire =
    obj && typeof obj.wire_type === "string" ? obj.wire_type : "LIST";
  const count =
    obj && typeof obj.count === "number"
      ? (obj.count as number)
      : (items?.length ?? 0);

  const srcs = (items ?? [])
    .map((it) => resolveSrc(it, executionId))
    .filter((s): s is string => s !== null);

  // Old logs (no per-tile assets) or nothing renderable → compact count badge.
  if (srcs.length === 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400">
        <span className="text-purple-400">{wire}</span>
        <span>{count} items</span>
      </span>
    );
  }

  return (
    <div>
      <div className="mb-1 text-[9px] text-gray-500">
        {wire} · {count} {count === 1 ? "tile" : "tiles"}
        {srcs.length < count && ` (showing ${srcs.length})`}
      </div>
      <div className="flex flex-wrap gap-1">
        {srcs.map((src, i) => (
          <Tile key={i} src={src} alt={`${label || "tile"}[${i}]`} />
        ))}
      </div>
    </div>
  );
}
