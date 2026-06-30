/** Renders TEXT wire type and large_string __type markers.
 * - All text shown in a readable box by default (expanded)
 * - Collapse toggle for long text
 * - Copy button
 */

import { useState } from "react";
import type { ValueRendererProps } from "./registry";

export default function TextRenderer({ value }: ValueRendererProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (value === null || value === undefined) {
    return <span className="text-gray-600 text-[10px]">—</span>;
  }

  // large_string __type marker (truncated server-side)
  if (typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    if (obj.__type === "large_string") {
      const preview = String(obj.preview || "");
      return (
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-gray-500 text-[10px]">
              [{String(obj.length)} chars, truncated]
            </span>
            <button
              onClick={() => navigator.clipboard.writeText(preview)}
              className="text-gray-500 hover:text-gray-300 text-[10px]"
            >
              copy preview
            </button>
          </div>
          <pre className="text-[10px] text-gray-300 bg-gray-800 p-1.5 rounded overflow-x-auto max-h-40 overflow-y-auto whitespace-pre-wrap">
            {preview}...
          </pre>
        </div>
      );
    }
  }

  const text = String(value);
  const lineCount = text.split("\n").length;
  const isLong = text.length > 300;

  // Short single-line text: inline display
  if (text.length <= 80 && lineCount === 1) {
    return <span className="text-green-400 text-[11px]">{text}</span>;
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-0.5">
        {isLong && (
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="text-blue-400 hover:text-blue-300 text-[10px] underline"
          >
            {collapsed ? `expand (${lineCount} lines)` : "collapse"}
          </button>
        )}
        <button
          onClick={() => navigator.clipboard.writeText(text)}
          className="text-gray-500 hover:text-gray-300 text-[10px]"
        >
          copy
        </button>
      </div>
      {collapsed ? (
        <span className="text-green-400 text-[11px]">
          {text.slice(0, 100)}...
        </span>
      ) : (
        <pre className="text-[10px] text-gray-300 bg-gray-800 p-1.5 rounded overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap">
          {text}
        </pre>
      )}
    </div>
  );
}
