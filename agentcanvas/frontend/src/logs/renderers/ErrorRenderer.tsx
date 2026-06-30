/** Renders error messages with red styling and collapsible stack trace. */

import { useState } from "react";
import clsx from "clsx";
import type { ValueRendererProps } from "./registry";

export default function ErrorRenderer({ value }: ValueRendererProps) {
  const [expanded, setExpanded] = useState(false);
  const text = String(value || "");
  const lines = text.split("\n");
  const isMultiline = lines.length > 3;

  return (
    <div className="rounded border border-red-800/50 bg-red-950/30 px-2 py-1.5">
      <div className="flex items-center gap-1.5 text-[11px]">
        <span className="text-red-400 font-medium">Error</span>
        {isMultiline && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-red-500 hover:text-red-300 text-[10px] underline"
          >
            {expanded ? "collapse" : `${lines.length} lines`}
          </button>
        )}
      </div>
      <pre
        className={clsx(
          "mt-0.5 text-[10px] text-red-300 font-mono whitespace-pre-wrap",
          !expanded && isMultiline && "max-h-12 overflow-hidden",
        )}
      >
        {text}
      </pre>
    </div>
  );
}
