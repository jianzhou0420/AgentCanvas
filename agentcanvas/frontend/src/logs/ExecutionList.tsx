/** Execution list sidebar — shows recent canvas + eval runs. */

import { useEffect, useState } from "react";
import clsx from "clsx";
import { logApi } from "./logApi";
import type { ExecutionListItem } from "./types";

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
  activeExecutionId?: string | null;
}

export default function ExecutionList({
  selectedId,
  onSelect,
  activeExecutionId,
}: Props) {
  const [executions, setExecutions] = useState<ExecutionListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const data = await logApi.listExecutions();
      setExecutions(data.executions);
    } catch {
      // ignore
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, 10_000);
    return () => clearInterval(interval);
  }, []);

  if (loading && executions.length === 0) {
    return (
      <div className="p-3 text-gray-600 text-xs">Loading executions...</div>
    );
  }

  if (executions.length === 0) {
    return (
      <div className="p-3 text-gray-600 text-xs">No executions found.</div>
    );
  }

  return (
    <div className="flex flex-col overflow-y-auto">
      {executions.map((ex) => {
        const isActive = ex.execution_id === activeExecutionId;
        const isSelected = ex.execution_id === selectedId;
        const date = new Date(ex.modified * 1000);

        return (
          <button
            key={ex.execution_id}
            onClick={() => onSelect(ex.execution_id)}
            className={clsx(
              "flex flex-col gap-0.5 px-3 py-2 text-left border-b border-gray-800 transition",
              isSelected
                ? "bg-blue-600/20 border-l-2 border-l-blue-500"
                : "hover:bg-gray-800/50",
            )}
          >
            <div className="flex items-center gap-1.5">
              {isActive && (
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse shrink-0" />
              )}
              <span className="text-[11px] text-gray-300 font-mono truncate">
                {ex.execution_id.slice(0, 12)}
              </span>
              <span
                className={clsx(
                  "ml-auto rounded px-1 py-0.5 text-[9px] font-medium shrink-0",
                  ex.source === "canvas"
                    ? "bg-blue-600/30 text-blue-300"
                    : "bg-purple-600/30 text-purple-300",
                )}
              >
                {ex.source}
              </span>
            </div>
            <div className="text-[10px] text-gray-500">
              {date.toLocaleDateString()} {date.toLocaleTimeString()}
            </div>
          </button>
        );
      })}
    </div>
  );
}
