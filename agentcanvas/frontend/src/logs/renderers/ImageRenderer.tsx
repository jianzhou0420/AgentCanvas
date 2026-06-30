/** Renders IMAGE/DEPTH/OBSERVATION values.
 * - Asset refs (__type === "asset" / "asset_group") → URL-based <img>
 * - Base64 strings → <img> with click-to-zoom
 * - __type markers (ndarray with shape) → shape summary
 * - OBSERVATION dicts → RGB + depth side by side
 */

import { useState, useContext } from "react";
import type { ValueRendererProps } from "./registry";
import { LogContext } from "../LogContext";
import { logApi } from "../logApi";

function ImageDisplay({
  src,
  alt,
  large,
}: {
  src: string;
  alt: string;
  large?: boolean;
}) {
  const [zoomed, setZoomed] = useState(false);
  const resolvedSrc = src.startsWith("data:")
    ? src
    : src.startsWith("http") || src.startsWith("/")
      ? src
      : `data:image/jpeg;base64,${src}`;

  return (
    <>
      <img
        src={resolvedSrc}
        alt={alt}
        className={
          large
            ? "max-h-80 max-w-full cursor-pointer rounded border border-gray-700 object-contain"
            : "max-h-32 max-w-48 cursor-pointer rounded border border-gray-700 object-contain"
        }
        onClick={() => setZoomed(true)}
      />
      {zoomed && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setZoomed(false)}
        >
          <img
            src={resolvedSrc}
            alt={alt}
            className="max-h-[90vh] max-w-[90vw] rounded"
          />
        </div>
      )}
    </>
  );
}

export default function ImageRenderer({ value, label }: ValueRendererProps) {
  const { executionId, viewMode } = useContext(LogContext);
  const large = viewMode === "detail";

  if (value === null || value === undefined) {
    return <span className="text-gray-600 text-[10px]">no image</span>;
  }

  // Asset reference from sidecar storage
  if (typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;

    // Single asset file
    if (obj.__type === "asset" && typeof obj.path === "string" && executionId) {
      const url = logApi.assetUrl(executionId, obj.path);
      return <ImageDisplay src={url} alt={label || "image"} large={large} />;
    }

    // Asset group (OBSERVATION composite)
    if (obj.__type === "asset_group") {
      const rgbAsset =
        obj.rgb && typeof obj.rgb === "object"
          ? (obj.rgb as Record<string, unknown>)
          : null;
      const depthAsset =
        obj.depth && typeof obj.depth === "object"
          ? (obj.depth as Record<string, unknown>)
          : null;
      return (
        <div className="flex gap-2">
          {rgbAsset?.__type === "asset" && executionId && (
            <div>
              <div className="text-[9px] text-gray-500 mb-0.5">RGB</div>
              <ImageDisplay
                src={logApi.assetUrl(executionId, String(rgbAsset.path))}
                alt="rgb"
                large={large}
              />
            </div>
          )}
          {depthAsset?.__type === "asset" && executionId && (
            <div>
              <div className="text-[9px] text-gray-500 mb-0.5">Depth</div>
              <ImageDisplay
                src={logApi.assetUrl(executionId, String(depthAsset.path))}
                alt="depth"
                large={large}
              />
            </div>
          )}
        </div>
      );
    }
  }

  // Base64 string
  if (typeof value === "string" && value.length > 100) {
    return <ImageDisplay src={value} alt={label || "image"} large={large} />;
  }

  // Object with __type (serialized array)
  if (typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;

    // OBSERVATION composite: {rgb, depth}
    if (obj.rgb !== undefined || obj.depth !== undefined) {
      return (
        <div className="flex gap-2">
          {obj.rgb != null && (
            <div>
              <div className="text-[9px] text-gray-500 mb-0.5">RGB</div>
              <ImageRenderer value={obj.rgb} label="rgb" />
            </div>
          )}
          {obj.depth != null && (
            <div>
              <div className="text-[9px] text-gray-500 mb-0.5">Depth</div>
              <ImageRenderer value={obj.depth} label="depth" />
            </div>
          )}
        </div>
      );
    }

    // __type summary (ndarray shape)
    if (obj.__type) {
      return (
        <span className="inline-flex items-center gap-1 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400">
          <span className="text-purple-400">{String(obj.__type)}</span>
          {obj.shape != null && <span>[{String(obj.shape)}]</span>}
          {obj.dtype != null && (
            <span className="text-gray-500">{String(obj.dtype)}</span>
          )}
        </span>
      );
    }
  }

  // Fallback
  return (
    <span className="text-gray-500 text-[10px]">
      {String(value).slice(0, 100)}
    </span>
  );
}
