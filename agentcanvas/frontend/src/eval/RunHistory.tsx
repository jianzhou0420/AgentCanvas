import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import clsx from "clsx";
import { evalApi } from "./evalApi";
import type { EvalRunSummary, EvalStatus } from "./types";

interface Props {
  onSelectRun: (run: EvalRunSummary) => void;
  // While a run is active, poll so a newly-started run appears and updates to
  // its terminal state without a manual Refresh.
  active?: boolean;
}

function statusColor(status: EvalStatus): string {
  switch (status) {
    case "completed":
      return "text-green-400";
    case "running":
    case "pending":
      return "text-blue-400";
    case "cancelled":
      return "text-orange-400";
    case "error":
      return "text-red-400";
    default:
      return "text-gray-500";
  }
}

export default function RunHistory({ onSelectRun, active }: Props) {
  const [runs, setRuns] = useState<EvalRunSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function load() {
    setLoading(true);
    evalApi
      .listRuns()
      .then(({ runs: r }) => setRuns(r || []))
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
    if (!active) return;
    // Re-runs when `active` flips false → one final load catches the just-
    // finished run's terminal state after the scheduler reaps it.
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  async function handleDelete(runId: string, e: React.MouseEvent) {
    e.stopPropagation();
    await evalApi.deleteRun(runId);
    setRuns((prev) => prev.filter((r) => r.run_id !== runId));
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Run History
        </h3>
        <button
          onClick={load}
          className="text-xs text-gray-500 hover:text-gray-300"
        >
          Refresh
        </button>
      </div>

      {/* Only on the first load (empty list). Showing it on every 3s poll
          inserts/removes this line and makes the run list jump. */}
      {loading && runs.length === 0 && (
        <div className="text-xs text-gray-500">Loading...</div>
      )}
      {error && <div className="text-xs text-red-400">{error}</div>}

      {!loading && runs.length === 0 && (
        <div className="text-xs text-gray-500">No past runs</div>
      )}

      <div className="flex flex-col gap-1">
        {runs.map((run) => {
          const spl = run.aggregate_metrics?.spl;
          const date = new Date(run.created_at).toLocaleDateString();
          return (
            <div
              key={run.run_id}
              onClick={() => onSelectRun(run)}
              className="flex cursor-pointer items-center justify-between rounded border border-gray-800 bg-gray-800/30 px-2 py-1.5 hover:bg-gray-800/70"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs text-gray-300">
                  {run.graph_name}
                </div>
                <div className="flex items-center gap-1.5 text-xs">
                  <span className={statusColor(run.status)}>{run.status}</span>
                  <span className="text-gray-600">·</span>
                  <span className="text-gray-500">{date}</span>
                  {spl !== undefined && (
                    <>
                      <span className="text-gray-600">·</span>
                      <span className="font-mono text-gray-400">
                        SPL {spl.toFixed(3)}
                      </span>
                    </>
                  )}
                </div>
              </div>
              <button
                onClick={(e) => handleDelete(run.run_id, e)}
                className={clsx(
                  "ml-1 shrink-0 rounded p-0.5 text-gray-600 hover:text-red-400",
                )}
                title="Delete run"
              >
                <Trash2 size={12} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
