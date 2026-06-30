import { useEffect, useState, useCallback } from "react";
import { Play, Square, Download, Loader2 } from "lucide-react";
import clsx from "clsx";
import { evalApi } from "./evalApi";
import GraphSelector from "./GraphSelector";
import EnvInfoPanel from "./EnvInfoPanel";
import EvalProgressBar from "./EvalProgressBar";
import MetricCards from "./MetricCards";
import EpisodesTable from "./EpisodesTable";
import RunHistory from "./RunHistory";
import type {
  GraphIntrospection,
  EvalRunSummary,
  EvalEpisodeResult,
  EvalStatus,
} from "./types";

const inputCls =
  "mt-1 w-full bg-gray-800 text-gray-200 text-sm px-2 py-1 rounded border border-gray-700";

// ── Config panel (left sidebar) ──

interface ConfigPanelProps {
  isRunning: boolean;
  onStart: () => void;
  onStop: () => void;
  graphName: string;
  setGraphName: (v: string) => void;
  introspection: GraphIntrospection | null;
  setIntrospection: (v: GraphIntrospection | null) => void;
  selectors: Record<string, string | number>;
  setSelectors: (v: Record<string, string | number>) => void;
  episodeCount: number;
  setEpisodeCount: (v: number) => void;
  stepBudget: number;
  setStepBudget: (v: number) => void;
  startEpisodeIndex: number;
  setStartEpisodeIndex: (v: number) => void;
  workerCount: number;
  setWorkerCount: (v: number) => void;
}

function ConfigPanel({
  isRunning,
  onStart,
  onStop,
  graphName,
  setGraphName,
  introspection,
  setIntrospection,
  selectors,
  setSelectors,
  episodeCount,
  setEpisodeCount,
  stepBudget,
  setStepBudget,
  startEpisodeIndex,
  setStartEpisodeIndex,
  workerCount,
  setWorkerCount,
}: ConfigPanelProps) {
  const canStart = !isRunning && graphName.length > 0;

  async function handleLoadNodeset() {
    if (!introspection?.env_nodeset) return;
    try {
      const apiBase = import.meta.env.VITE_API_URL || "";
      await fetch(
        `${apiBase}/api/components/nodesets/${encodeURIComponent(introspection.env_nodeset)}/load`,
        { method: "POST" },
      );
      const updated = await evalApi.introspectGraph(graphName);
      setIntrospection(updated);
    } catch {
      // silently fail
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
        Graph Evaluation
      </h3>

      <GraphSelector
        value={graphName}
        onChange={(name, intro) => {
          setGraphName(name);
          setIntrospection(intro);
          // Reset selectors on graph swap so the new env's env panel seeds
          // them from its on_load() defaults (handled inside EnvInfoPanel).
          setSelectors({});
          if (intro?.metadata?.step_budget)
            setStepBudget(intro.metadata.step_budget);
        }}
        disabled={isRunning}
      />

      <EnvInfoPanel
        introspection={introspection}
        selectors={selectors}
        episodeCount={episodeCount}
        startEpisodeIndex={startEpisodeIndex}
        disabled={isRunning}
        onSelectorsChange={setSelectors}
        onEpisodeCountChange={setEpisodeCount}
        onStartEpisodeChange={setStartEpisodeIndex}
        onLoadNodeset={handleLoadNodeset}
      />

      <label className="text-xs text-gray-400">
        Step Budget
        <input
          type="number"
          min={1}
          value={stepBudget}
          onChange={(e) => setStepBudget(Number(e.target.value))}
          disabled={isRunning}
          className={inputCls}
        />
      </label>

      <label className="text-xs text-gray-400">
        Worker Count
        <input
          type="number"
          min={1}
          value={workerCount}
          onChange={(e) => setWorkerCount(Number(e.target.value))}
          disabled={isRunning}
          className={inputCls}
        />
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={onStart}
          disabled={!canStart}
          className={clsx(
            "flex items-center justify-center gap-1.5 rounded px-3 py-2 text-sm font-medium",
            canStart
              ? "bg-green-600 text-white hover:bg-green-500"
              : "cursor-not-allowed bg-gray-800 text-gray-600",
          )}
        >
          <Play size={14} /> Start
        </button>
        <button
          onClick={onStop}
          disabled={!isRunning}
          className={clsx(
            "flex items-center justify-center gap-1.5 rounded px-3 py-2 text-sm font-medium",
            isRunning
              ? "bg-red-600 text-white hover:bg-red-500"
              : "cursor-not-allowed bg-gray-800 text-gray-600",
          )}
        >
          <Square size={14} /> Stop
        </button>
      </div>
    </div>
  );
}

// ── Per-worker progress chip row (ADR-028 PB-3) ──

function PerWorkerProgress({ episodes }: { episodes: EvalEpisodeResult[] }) {
  // Group completed episodes by worker_id. Hide entirely at single-worker
  // (default behaviour identical to pre-PB-3).
  const counts = new Map<number, number>();
  for (const ep of episodes) {
    if (ep.worker_id === undefined) continue;
    counts.set(ep.worker_id, (counts.get(ep.worker_id) ?? 0) + 1);
  }
  if (counts.size <= 1) return null;
  const entries = [...counts.entries()].sort((a, b) => a[0] - b[0]);
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400">
      <span className="text-gray-500">Per worker:</span>
      {entries.map(([workerId, n]) => (
        <span
          key={workerId}
          className="rounded border border-gray-700 bg-gray-800 px-2 py-0.5 font-mono"
        >
          W{workerId}: {n}
        </span>
      ))}
    </div>
  );
}

// ── Main EvalPage ──

export default function EvalPage() {
  // Config state
  const [graphName, setGraphName] = useState("");
  const [introspection, setIntrospection] = useState<GraphIntrospection | null>(
    null,
  );
  const [selectors, setSelectors] = useState<Record<string, string | number>>(
    {},
  );
  const [episodeCount, setEpisodeCount] = useState(5);
  const [stepBudget, setStepBudget] = useState(500);
  const [startEpisodeIndex, setStartEpisodeIndex] = useState(0);
  const [workerCount, setWorkerCount] = useState(1);

  // Run state
  const [status, setStatus] = useState<EvalStatus>("none");
  const [run, setRun] = useState<EvalRunSummary | null>(null);
  const [episodes, setEpisodes] = useState<EvalEpisodeResult[]>([]);
  const [actionError, setActionError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  const isRunning = status === "running" || status === "pending";

  const poll = useCallback(async () => {
    try {
      const { status: s, run: r } = await evalApi.getStatus();
      setStatus(s as EvalStatus);
      setRun(r);
      if (r && (s === "running" || s === "completed")) {
        const { episodes: eps } = await evalApi.getEpisodes();
        setEpisodes(eps);
      }
    } catch {
      // ignore transient errors
    }
  }, []);

  useEffect(() => {
    poll();
  }, [poll]);

  useEffect(() => {
    if (!isRunning) return;
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [isRunning, poll]);

  async function handleStart() {
    setActionError(null);
    try {
      // Pass an empty `split` so the backend's legacy default ("val_unseen")
      // doesn't sneak into the cascade. Everything the env panel needs is
      // already in `selectors` (which itself was seeded from the env panel's
      // on_load() so it carries the correct field names per env).
      await evalApi.startEval({
        graph_name: graphName,
        split: "",
        selectors,
        episode_count: episodeCount,
        step_budget: stepBudget,
        start_episode_index: startEpisodeIndex,
        worker_count: workerCount,
      });
      await poll();
    } catch (e: unknown) {
      setActionError(String(e));
    }
  }

  async function handleStop() {
    setActionError(null);
    try {
      await evalApi.stopEval();
      await poll();
    } catch (e: unknown) {
      setActionError(String(e));
    }
  }

  async function handleExport() {
    if (!run) return;
    setExporting(true);
    try {
      const data = await evalApi.exportRun(run.run_id);
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `eval_${run.run_id}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // silently fail
    } finally {
      setExporting(false);
    }
  }

  function handleSelectRun(selectedRun: EvalRunSummary) {
    setRun(selectedRun);
    setStatus(selectedRun.status);
    evalApi
      .getEpisodes()
      .then(({ episodes: eps }) => setEpisodes(eps))
      .catch(() => {});
  }

  const hasResults = run && run.status !== "none" && run.status !== "pending";

  return (
    <div className="flex min-h-0 flex-1 gap-1 p-1">
      {/* Left sidebar */}
      <div className="flex w-[300px] flex-shrink-0 flex-col gap-4 overflow-auto rounded border border-gray-800 bg-gray-900 p-3">
        <ConfigPanel
          isRunning={isRunning}
          onStart={handleStart}
          onStop={handleStop}
          graphName={graphName}
          setGraphName={setGraphName}
          introspection={introspection}
          setIntrospection={setIntrospection}
          selectors={selectors}
          setSelectors={setSelectors}
          episodeCount={episodeCount}
          setEpisodeCount={setEpisodeCount}
          stepBudget={stepBudget}
          setStepBudget={setStepBudget}
          startEpisodeIndex={startEpisodeIndex}
          setStartEpisodeIndex={setStartEpisodeIndex}
          workerCount={workerCount}
          setWorkerCount={setWorkerCount}
        />

        {actionError && (
          <div className="rounded border border-red-800 bg-red-900/20 px-2 py-1 text-xs text-red-400">
            {actionError}
          </div>
        )}

        <div className="border-t border-gray-800" />

        <RunHistory onSelectRun={handleSelectRun} />
      </div>

      {/* Right main area */}
      <div className="flex min-h-0 flex-1 flex-col gap-1">
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto rounded border border-gray-800 bg-gray-900 p-3">
          {run && (
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span>
                Status:{" "}
                <span
                  className={clsx(
                    "font-medium",
                    run.status === "running" && "text-yellow-400",
                    run.status === "completed" && "text-green-400",
                    run.status === "cancelled" && "text-orange-400",
                    run.status === "error" && "text-red-400",
                    run.status === "pending" && "text-blue-400",
                  )}
                >
                  {run.status}
                </span>
              </span>
              {isRunning && (
                <Loader2 size={12} className="animate-spin text-blue-400" />
              )}
              <span>Elapsed: {run.elapsed_sec.toFixed(1)}s</span>
            </div>
          )}

          <EvalProgressBar
            completed_count={run?.completed_count ?? 0}
            total_episodes={run?.total_episodes ?? run?.episode_count ?? 0}
            status={status}
          />

          <PerWorkerProgress episodes={episodes} />

          <div>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-400">
              Aggregate Metrics
            </h3>
            <MetricCards aggregate_metrics={run?.aggregate_metrics ?? {}} />
          </div>

          <div>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-400">
              Episodes
            </h3>
            <EpisodesTable episodes={episodes} />
          </div>

          <div className="flex justify-end">
            <button
              onClick={handleExport}
              disabled={!hasResults || exporting}
              className={clsx(
                "flex items-center gap-1.5 rounded px-3 py-1.5 text-sm",
                hasResults && !exporting
                  ? "bg-gray-700 text-gray-200 hover:bg-gray-600"
                  : "cursor-not-allowed bg-gray-800 text-gray-600",
              )}
            >
              {exporting ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Download size={14} />
              )}
              Export JSON
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
