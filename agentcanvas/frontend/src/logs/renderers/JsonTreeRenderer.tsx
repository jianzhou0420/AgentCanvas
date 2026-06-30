/** Fallback renderer — collapsible JSON tree with syntax coloring. */

import { useState } from "react";
import type { ValueRendererProps } from "./registry";

export default function JsonTreeRenderer({ value }: ValueRendererProps) {
  const [expanded, setExpanded] = useState(false);

  if (value === null || value === undefined) {
    return <span className="text-gray-500 text-[10px]">null</span>;
  }

  const json = JSON.stringify(value, null, 2);
  const isLong = json.length > 120;
  const keyCount =
    typeof value === "object" && !Array.isArray(value)
      ? Object.keys(value as Record<string, unknown>).length
      : Array.isArray(value)
        ? value.length
        : 0;

  if (!isLong) {
    return (
      <code className="text-[10px] text-gray-300 bg-gray-800 px-1 py-0.5 rounded">
        {json}
      </code>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-blue-400 hover:text-blue-300 text-[10px] underline"
        >
          {expanded ? "collapse" : "expand"}
        </button>
        <span className="text-gray-600 text-[10px]">
          {Array.isArray(value) ? `[${keyCount} items]` : `{${keyCount} keys}`}
        </span>
        <button
          onClick={() => navigator.clipboard.writeText(json)}
          className="text-gray-500 hover:text-gray-300 text-[10px]"
        >
          copy
        </button>
      </div>
      {expanded && (
        <pre className="mt-0.5 text-[10px] text-gray-300 bg-gray-800 p-1.5 rounded overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap">
          {json}
        </pre>
      )}
    </div>
  );
}
