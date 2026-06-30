/** Summary bar — aggregate stats for the selected execution. */

import { useEffect, useState } from "react";
import { logApi } from "./logApi";
import type { ExecutionSummary } from "./types";

function formatDuration(
  startStr: string | null,
  endStr: string | null,
): string {
  if (!startStr) return "—";
  const start = new Date(startStr).getTime();
  const end = endStr ? new Date(endStr).getTime() : Date.now();
  const ms = end - start;
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

interface Props {
  executionId: string;
  isLive: boolean;
}

export default function LogSummaryBar({ executionId, isLive }: Props) {
  const [summary, setSummary] = useState<ExecutionSummary | null>(null);

  useEffect(() => {
    let cancelled = false;

    logApi
      .getSummary(executionId)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch(() => {});

    if (isLive) {
      const interval = setInterval(() => {
        logApi
          .getSummary(executionId)
          .then((s) => {
            if (!cancelled) setSummary(s);
          })
          .catch(() => {});
      }, 3000);
      return () => {
        cancelled = true;
        clearInterval(interval);
      };
    }

    return () => {
      cancelled = true;
    };
  }, [executionId, isLive]);

  if (!summary) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-800 text-[11px]">
      {isLive && (
        <span className="flex items-center gap-1 text-green-400 font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
          Live
        </span>
      )}
      <span className="text-gray-400">
        <span className="text-gray-200 font-medium">{summary.total_steps}</span>{" "}
        steps
      </span>
      <span className="text-gray-400">
        <span className="text-gray-200 font-medium">
          {summary.total_firings}
        </span>{" "}
        firings
      </span>
      {summary.error_count > 0 && (
        <span className="text-red-400">
          <span className="font-medium">{summary.error_count}</span> errors
        </span>
      )}
      <span className="text-gray-500">
        {formatDuration(summary.started_at, summary.ended_at)}
      </span>
      <span className="text-gray-600 text-[10px] truncate">
        {summary.node_types_fired.length} node types
      </span>
    </div>
  );
}
