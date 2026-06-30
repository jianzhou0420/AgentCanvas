/** Log list display field for viewer nodes.
 * Renders navigation actions or LLM reasoning steps as a scrollable list. */

import clsx from "clsx";
import type { DisplayFieldProps } from "../viewerFieldRegistry";

const STEP_BORDER: Record<string, string> = {
  reasoning: "border-l-gray-500",
  tool_call: "border-l-blue-500",
  tool_result: "border-l-green-500",
  decision: "border-l-yellow-500",
};

const ACTION_COLOR: Record<number, string> = {
  0: "bg-red-900/50 text-red-300",
  1: "bg-green-900/50 text-green-300",
  2: "bg-yellow-900/50 text-yellow-300",
  3: "bg-purple-900/50 text-purple-300",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderLogEntry(entry: any, index: number): React.ReactNode {
  // Navigation action format: { action, action_name, step?, done? }
  if (entry.action_name !== undefined) {
    return (
      <div
        key={entry.step ?? index}
        className="flex items-center gap-1.5 text-[10px]"
      >
        <span className="w-4 text-right font-mono text-gray-500">
          {entry.step ?? index}
        </span>
        <span
          className={clsx(
            "rounded px-1 py-0.5 font-mono font-medium",
            ACTION_COLOR[entry.action],
          )}
        >
          {entry.action_name}
        </span>
        {entry.done && <span className="text-red-400">DONE</span>}
      </div>
    );
  }
  // LLM step format: { type, tool?, content }
  return (
    <div
      key={index}
      className={clsx(
        "border-l-2 py-0.5 pl-1.5 text-[10px]",
        STEP_BORDER[entry.type] || "border-l-gray-700",
      )}
    >
      <span className="font-medium text-gray-400">{entry.type}</span>
      {entry.tool && <span className="ml-1 text-blue-400">{entry.tool}</span>}
      <div className="line-clamp-2 text-gray-300">{entry.content}</div>
    </div>
  );
}

export default function LogListField({ value, label }: DisplayFieldProps) {
  const allEntries: unknown[] = Array.isArray(value) ? value : [];

  return (
    <div>
      {label && <div className="mb-0.5 text-[9px] text-gray-500">{label}</div>}
      {allEntries.length === 0 ? (
        <div className="italic text-gray-600 text-[10px]">
          Waiting for data...
        </div>
      ) : (
        <div className="max-h-60 space-y-0.5 overflow-auto">
          {allEntries.map((entry, i) => renderLogEntry(entry, i))}
        </div>
      )}
    </div>
  );
}
