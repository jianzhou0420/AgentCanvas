/**
 * MonitorPage — system resource sparklines (CPU / memory / per-GPU)
 * plus the JobScheduler queue (ADR-eval-003).
 *
 * Polls /api/system/usage and /api/eval/v2/queue every 1s. For each
 * running job, fetches /api/eval/v2/runs/{run_id} so we can show
 * episodes-done / total. Queued jobs only show their resource
 * declaration — the run subprocess hasn't spawned yet so there is
 * no per-episode state to read.
 */
import { useEffect, useRef, useState } from "react";
import Sparkline from "./Sparkline";
import NodeTimingTable from "./NodeTimingTable";

const HISTORY_LEN = 60; // 60 samples × 1s = 1 minute window
const POLL_MS = 1000;

interface GpuUsage {
  index: number;
  name: string;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  mem_pct: number;
}

interface SystemUsage {
  cpu_pct: number;
  cpu_count: number;
  mem_used_mb: number;
  mem_total_mb: number;
  mem_pct: number;
  gpus: GpuUsage[];
  event_loop_lag_ms?: number;
  gpu_procs?: { pid: number; mem_mb: number; owner: string }[];
}

interface QueuedJob {
  run_id: string;
  marginal_vram_mb: number;
  exclusive_gpu: boolean;
  priority: string;
  submitted_at: string;
}

interface RunningJob {
  run_id: string;
  pid: number;
  marginal_vram_mb: number;
  exclusive_gpu: boolean;
  started_at: string;
  cancel_requested: boolean;
}

interface QueueState {
  queued: QueuedJob[];
  running: RunningJob[];
  usable_vram_mb: number;
  reserved_vram_mb: number;
}

interface RunSummary {
  run_id: string;
  status: string;
  graph_name?: string;
  config?: { episode_count?: number; graph_name?: string };
  episodes?: Array<{ episode_index: number; metrics?: Record<string, number> }>;
  episodes_done?: number;
  episodes_total?: number;
  scheduler_state?: string;
  started_at?: string;
  finished_at?: string;
  error?: string;
}

interface RunListItem {
  run_id: string;
  source: "eval" | "canvas";
  graph_name?: string | null;
  status?: string | null;
  started?: string | null;
  finished?: string | null;
  episode_count?: number | null;
}

interface NodeTiming {
  node_type: string;
  count: number;
  compute_ms: {
    mean: number;
    p5: number;
    p50: number;
    p95: number;
    total: number;
  };
  queue_wait_ms: { mean: number | null; p95: number | null };
  transport_ms?: {
    total: number;
    mean: number;
    p5: number;
    p95: number;
  } | null;
  transfer_bytes?: number;
  share_pct: number;
}

interface RunDetail {
  run_id: string;
  source: "eval" | "canvas";
  node_timing: NodeTiming[];
  totals: {
    firings: number;
    compute_ms: number;
    tokens: number;
    usd_cost: number;
    llm_calls: number;
    transport_ms?: number;
    transfer_bytes?: number;
  };
  resources: SystemUsage[];
  eval?: {
    config?: { graph_name?: string };
    status?: string;
    aggregate_metrics?: Record<string, number>;
    total_episodes?: number;
  } | null;
}

function shortId(id: string): string {
  return id.length > 12 ? id.slice(0, 8) + "…" : id;
}

function fmtTime(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString();
}

function fmtElapsed(iso?: string): string {
  if (!iso) return "—";
  const start = new Date(iso).getTime();
  if (Number.isNaN(start)) return "—";
  const sec = Math.max(0, Math.round((Date.now() - start) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m${s.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h${(m % 60).toString().padStart(2, "0")}m`;
}

export default function MonitorPage() {
  const [usage, setUsage] = useState<SystemUsage | null>(null);
  const [queue, setQueue] = useState<QueueState | null>(null);
  const [runs, setRuns] = useState<Record<string, RunSummary>>({});
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string>("");

  const cpuHist = useRef<number[]>([]);
  const memHist = useRef<number[]>([]);
  const gpuHist = useRef<Record<number, { util: number[]; mem: number[] }>>({});
  // tick is just to trigger re-render after we mutate the refs
  const [, setTick] = useState(0);

  // Live ↔ Run view (run selection kept local — the established pattern across
  // the Replay/Log pages; not in the global store).
  const [mode, setMode] = useState<"live" | "run">("live");
  const [runList, setRunList] = useState<RunListItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runErr, setRunErr] = useState<string>("");

  useEffect(() => {
    if (mode !== "live") return; // Run mode gates the 1s machine/scheduler poll
    let cancelled = false;
    const tick = async () => {
      try {
        const [u, q, r] = await Promise.all([
          fetch("/api/system/usage").then((res) => res.json()),
          fetch("/api/eval/v2/queue").then((res) => res.json()),
          fetch("/api/eval/v2/runs").then((res) => res.json()),
        ]);
        if (cancelled) return;
        setError("");

        // Push into rolling buffers.
        const usg = u as SystemUsage;
        cpuHist.current = [...cpuHist.current, usg.cpu_pct].slice(-HISTORY_LEN);
        memHist.current = [...memHist.current, usg.mem_pct].slice(-HISTORY_LEN);
        for (const g of usg.gpus) {
          const prev = gpuHist.current[g.index] || { util: [], mem: [] };
          gpuHist.current[g.index] = {
            util: [...prev.util, g.util_pct].slice(-HISTORY_LEN),
            mem: [...prev.mem, g.mem_pct].slice(-HISTORY_LEN),
          };
        }
        setUsage(usg);
        setQueue(q as QueueState);

        // Fetch detail for every running job (queued jobs have no detail yet).
        const qs = q as QueueState;
        const detailPromises = qs.running.map((rj) =>
          fetch(`/api/eval/v2/runs/${rj.run_id}`)
            .then((res) => (res.ok ? res.json() : null))
            .catch(() => null),
        );
        const details = await Promise.all(detailPromises);
        if (cancelled) return;
        const next: Record<string, RunSummary> = {};
        details.forEach((d, i) => {
          if (d) next[qs.running[i].run_id] = d as RunSummary;
        });
        setRuns(next);

        // Recent finished runs — tail of /runs list (already sorted by backend).
        const allRuns = (r?.runs ?? r ?? []) as RunSummary[];
        const finished = allRuns
          .filter((x) => {
            const s = x.status || "";
            return s !== "running" && s !== "pending";
          })
          .slice(0, 8);
        setRecent(finished);

        setTick((t) => t + 1);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    // Backfill rolling buffers from persisted history so the Live sparklines
    // survive a page refresh (the buffers are otherwise in-memory only).
    const backfill = async () => {
      try {
        const r = await fetch(`/api/system/history?n=${HISTORY_LEN}`).then(
          (res) => res.json(),
        );
        const samples = (r?.samples ?? []) as SystemUsage[];
        if (cancelled || samples.length === 0) return;
        cpuHist.current = samples.map((s) => s.cpu_pct).slice(-HISTORY_LEN);
        memHist.current = samples.map((s) => s.mem_pct).slice(-HISTORY_LEN);
        const gh: Record<number, { util: number[]; mem: number[] }> = {};
        for (const s of samples) {
          for (const g of s.gpus ?? []) {
            const prev = gh[g.index] || { util: [], mem: [] };
            gh[g.index] = {
              util: [...prev.util, g.util_pct].slice(-HISTORY_LEN),
              mem: [...prev.mem, g.mem_pct].slice(-HISTORY_LEN),
            };
          }
        }
        gpuHist.current = gh;
        setTick((t) => t + 1);
      } catch {
        /* ignore — the live tick will populate the buffers */
      }
    };
    void backfill().then(() => void tick());
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [mode]);

  // Load the unified eval+canvas run list when entering Run mode.
  useEffect(() => {
    if (mode !== "run") return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/system/runs").then((res) => res.json());
        if (!cancelled) setRunList((r?.runs ?? []) as RunListItem[]);
      } catch (e) {
        if (!cancelled) setRunErr(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  // Load detail for the selected run (stale-guarded against fast switches).
  useEffect(() => {
    if (mode !== "run" || !selectedRunId) {
      setRunDetail(null);
      return;
    }
    let cancelled = false;
    const id = selectedRunId;
    setRunErr("");
    setRunDetail(null);
    (async () => {
      try {
        const res = await fetch(`/api/system/runs/${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error(`run ${id}: HTTP ${res.status}`);
        const d = (await res.json()) as RunDetail;
        if (!cancelled && id === selectedRunId) setRunDetail(d);
      } catch (e) {
        if (!cancelled) setRunErr(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode, selectedRunId]);

  const cancelRun = async (runId: string) => {
    try {
      await fetch(`/api/eval/v2/runs/${runId}/cancel`, { method: "POST" });
    } catch {
      /* swallow — next poll will reflect the result */
    }
  };

  const progressOf = (rj: RunningJob): { done: number; total: number } => {
    const r = runs[rj.run_id];
    if (!r) return { done: 0, total: 0 };
    const done = r.episodes_done ?? r.episodes?.length ?? 0;
    const total =
      r.episodes_total ?? r.config?.episode_count ?? r.episodes?.length ?? 0;
    return { done, total };
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-950 p-4 text-gray-200">
      {error && (
        <div className="mb-3 rounded border border-red-700 bg-red-900/30 px-3 py-2 text-xs text-red-300">
          {error}
        </div>
      )}

      {/* Mode toggle */}
      <div className="mb-4 flex items-center gap-1">
        {(["live", "run"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={
              "rounded px-3 py-1 text-xs font-medium capitalize " +
              (mode === m
                ? "bg-gray-200 text-gray-900"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700")
            }
          >
            {m}
          </button>
        ))}
        <span className="ml-2 text-[11px] text-gray-500">
          {mode === "live"
            ? "machine + scheduler · 1s"
            : "pick a past run (eval or canvas)"}
        </span>
      </div>

      {mode === "run" && (
        <RunView
          runList={runList}
          selectedRunId={selectedRunId}
          onSelect={setSelectedRunId}
          detail={runDetail}
          err={runErr}
        />
      )}

      {mode === "live" && (
        <>
          {/* Resources */}
          <section className="mb-6">
            <div className="mb-2 flex items-baseline justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">
                System resources
              </h2>
              <span className="text-xs text-gray-500">
                polling every {POLL_MS / 1000}s · {HISTORY_LEN}s window
              </span>
            </div>
            <div className="grid gap-3 lg:grid-cols-2">
              <ResourceCard
                title="CPU"
                valueLabel={
                  usage
                    ? `${usage.cpu_pct.toFixed(1)}% · ${usage.cpu_count} cores`
                    : "—"
                }
                values={cpuHist.current}
                stroke="#60a5fa"
                fill="rgba(96, 165, 250, 0.15)"
              />
              <ResourceCard
                title="Memory"
                valueLabel={
                  usage
                    ? `${(usage.mem_used_mb / 1024).toFixed(1)} / ${(
                        usage.mem_total_mb / 1024
                      ).toFixed(1)} GB · ${usage.mem_pct.toFixed(1)}%`
                    : "—"
                }
                values={memHist.current}
                stroke="#a78bfa"
                fill="rgba(167, 139, 250, 0.15)"
              />
              {usage?.gpus.map((g) => {
                const hist = gpuHist.current[g.index] || { util: [], mem: [] };
                return (
                  <div
                    key={g.index}
                    className="rounded border border-gray-800 bg-gray-900 p-3"
                  >
                    <div className="mb-1 flex items-baseline justify-between">
                      <div className="text-sm font-medium text-gray-200">
                        GPU {g.index} · {g.name}
                      </div>
                      <div className="text-xs text-gray-400">
                        {g.util_pct.toFixed(0)}% util ·{" "}
                        {(g.mem_used_mb / 1024).toFixed(1)} /{" "}
                        {(g.mem_total_mb / 1024).toFixed(1)} GB (
                        {g.mem_pct.toFixed(1)}%)
                      </div>
                    </div>
                    <Sparkline
                      values={hist.util}
                      height={48}
                      stroke="#34d399"
                      fill="rgba(52, 211, 153, 0.15)"
                    />
                    <Sparkline
                      values={hist.mem}
                      height={48}
                      stroke="#fbbf24"
                      fill="rgba(251, 191, 36, 0.15)"
                    />
                    <div className="mt-1 flex justify-between text-[10px] text-gray-500">
                      <span>util — green</span>
                      <span>vram — amber</span>
                    </div>
                  </div>
                );
              })}
              {usage && usage.gpus.length === 0 && (
                <div className="col-span-full rounded border border-gray-800 bg-gray-900 p-3 text-xs text-gray-500">
                  No GPUs detected (nvidia-smi unavailable).
                </div>
              )}
              {usage?.gpu_procs && usage.gpu_procs.length > 0 && (
                <div className="col-span-full rounded border border-gray-800 bg-gray-900 p-3">
                  <div className="mb-1.5 text-xs font-medium text-gray-300">
                    VRAM by process
                  </div>
                  <div className="space-y-1">
                    {usage.gpu_procs.map((p) => (
                      <div
                        key={p.pid}
                        className="flex items-center justify-between text-[11px] text-gray-400"
                      >
                        <span className="truncate">
                          <span className="text-gray-200">{p.owner}</span>{" "}
                          <span className="font-mono text-gray-600">
                            pid {p.pid}
                          </span>
                        </span>
                        <span className="tabular-nums">
                          {(p.mem_mb / 1024).toFixed(2)} GB
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Jobs */}
          <section>
            <div className="mb-2 flex items-baseline justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">
                JobScheduler
              </h2>
              {queue && (
                <span className="text-xs text-gray-500">
                  VRAM budget {queue.reserved_vram_mb} / {queue.usable_vram_mb}{" "}
                  MB admitted · {queue.running.length} running ·{" "}
                  {queue.queued.length} queued
                </span>
              )}
            </div>

            {/* Running */}
            <div className="mb-4 rounded border border-gray-800 bg-gray-900">
              <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                Running ({queue?.running.length ?? 0})
              </div>
              {queue && queue.running.length === 0 && (
                <div className="px-3 py-3 text-xs text-gray-500">
                  No active runs.
                </div>
              )}
              {queue?.running.map((rj) => {
                const detail = runs[rj.run_id];
                const { done, total } = progressOf(rj);
                const pct = total ? (done / total) * 100 : 0;
                const graph =
                  detail?.graph_name ||
                  detail?.config?.graph_name ||
                  "(loading…)";
                return (
                  <div
                    key={rj.run_id}
                    className="flex items-center gap-3 border-b border-gray-800 px-3 py-2 last:border-b-0"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="rounded bg-emerald-700/40 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
                          RUNNING
                        </span>
                        <span className="font-mono text-xs text-gray-400">
                          {shortId(rj.run_id)}
                        </span>
                        <span className="truncate text-sm text-gray-200">
                          {graph}
                        </span>
                        {rj.cancel_requested && (
                          <span className="rounded bg-yellow-800/40 px-1.5 py-0.5 text-[10px] text-yellow-300">
                            cancelling…
                          </span>
                        )}
                      </div>
                      <div className="mt-1 flex items-center gap-3 text-[11px] text-gray-500">
                        <span>pid {rj.pid}</span>
                        <span>vram {rj.marginal_vram_mb} MB</span>
                        {rj.exclusive_gpu && <span>exclusive-gpu</span>}
                        <span>elapsed {fmtElapsed(rj.started_at)}</span>
                      </div>
                      <div className="mt-1.5 flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded bg-gray-800">
                          <div
                            className="h-full rounded bg-emerald-500 transition-[width] duration-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-20 text-right text-xs tabular-nums text-gray-400">
                          {total ? `${done}/${total}` : `${done} ep`}
                        </span>
                      </div>
                    </div>
                    <button
                      className="rounded border border-red-700 px-2 py-1 text-xs text-red-300 hover:bg-red-900/30 disabled:opacity-50"
                      onClick={() => cancelRun(rj.run_id)}
                      disabled={rj.cancel_requested}
                    >
                      Cancel
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Queued */}
            <div className="mb-4 rounded border border-gray-800 bg-gray-900">
              <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                Queued ({queue?.queued.length ?? 0})
              </div>
              {queue && queue.queued.length === 0 && (
                <div className="px-3 py-3 text-xs text-gray-500">
                  Queue empty.
                </div>
              )}
              {queue?.queued.map((qj) => (
                <div
                  key={qj.run_id}
                  className="flex items-center gap-3 border-b border-gray-800 px-3 py-2 last:border-b-0"
                >
                  <span className="rounded bg-gray-700/60 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-gray-300">
                    {qj.priority}
                  </span>
                  <span className="font-mono text-xs text-gray-400">
                    {shortId(qj.run_id)}
                  </span>
                  <span className="text-[11px] text-gray-500">
                    vram {qj.marginal_vram_mb} MB
                    {qj.exclusive_gpu ? " · exclusive-gpu" : ""}
                  </span>
                  <span className="ml-auto text-[11px] text-gray-500">
                    submitted {fmtTime(qj.submitted_at)}
                  </span>
                  <button
                    className="rounded border border-red-700 px-2 py-1 text-xs text-red-300 hover:bg-red-900/30"
                    onClick={() => cancelRun(qj.run_id)}
                  >
                    Cancel
                  </button>
                </div>
              ))}
            </div>

            {/* Recent finished */}
            <div className="rounded border border-gray-800 bg-gray-900">
              <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                Recent ({recent.length})
              </div>
              {recent.length === 0 && (
                <div className="px-3 py-3 text-xs text-gray-500">
                  No finished runs yet.
                </div>
              )}
              {recent.map((r) => (
                <div
                  key={r.run_id}
                  className="flex items-center gap-3 border-b border-gray-800 px-3 py-2 last:border-b-0"
                >
                  <span
                    className={
                      "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase " +
                      (r.status === "done"
                        ? "bg-blue-800/40 text-blue-300"
                        : r.status === "error" || r.status === "aborted"
                          ? "bg-red-800/40 text-red-300"
                          : "bg-gray-700/60 text-gray-300")
                    }
                  >
                    {r.status}
                  </span>
                  <span className="font-mono text-xs text-gray-400">
                    {shortId(r.run_id)}
                  </span>
                  <span className="truncate text-sm text-gray-200">
                    {r.graph_name || r.config?.graph_name || "—"}
                  </span>
                  <span className="ml-auto text-[11px] text-gray-500">
                    {fmtTime(r.finished_at)}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

interface ResourceCardProps {
  title: string;
  valueLabel: string;
  values: number[];
  stroke: string;
  fill: string;
}

function ResourceCard({
  title,
  valueLabel,
  values,
  stroke,
  fill,
}: ResourceCardProps) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <div className="text-sm font-medium text-gray-200">{title}</div>
        <div className="text-xs text-gray-400">{valueLabel}</div>
      </div>
      <Sparkline values={values} height={64} stroke={stroke} fill={fill} />
    </div>
  );
}

function avgPeak(a: number[]): string {
  if (!a.length) return "—";
  const avg = a.reduce((s, x) => s + x, 0) / a.length;
  return `avg ${avg.toFixed(0)}% · peak ${Math.max(...a).toFixed(0)}%`;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 px-3 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-gray-500">
        {label}
      </div>
      <div className="tabular-nums text-gray-200">{value}</div>
    </div>
  );
}

function RunView({
  runList,
  selectedRunId,
  onSelect,
  detail,
  err,
}: {
  runList: RunListItem[];
  selectedRunId: string | null;
  onSelect: (id: string) => void;
  detail: RunDetail | null;
  err: string;
}) {
  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      {/* Run picker */}
      <div className="rounded border border-gray-800 bg-gray-900">
        <div className="border-b border-gray-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
          Runs ({runList.length})
        </div>
        <div className="max-h-[72vh] overflow-y-auto">
          {runList.length === 0 && (
            <div className="px-3 py-3 text-xs text-gray-500">
              No runs found.
            </div>
          )}
          {runList.map((r) => (
            <button
              key={`${r.source}:${r.run_id}`}
              onClick={() => onSelect(r.run_id)}
              className={
                "block w-full border-b border-gray-800 px-3 py-2 text-left last:border-b-0 hover:bg-gray-800/60 " +
                (selectedRunId === r.run_id ? "bg-gray-800" : "")
              }
            >
              <div className="flex items-center gap-2">
                <span
                  className={
                    "rounded px-1 py-0.5 text-[9px] font-medium uppercase " +
                    (r.source === "eval"
                      ? "bg-indigo-800/40 text-indigo-300"
                      : "bg-teal-800/40 text-teal-300")
                  }
                >
                  {r.source}
                </span>
                <span className="truncate text-xs text-gray-200">
                  {r.graph_name || r.run_id}
                </span>
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-[10px] text-gray-500">
                <span className="font-mono">{shortId(r.run_id)}</span>
                {r.status && <span>· {r.status}</span>}
                {r.episode_count != null && <span>· {r.episode_count} ep</span>}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Detail */}
      <div>
        {err && (
          <div className="mb-3 rounded border border-red-700 bg-red-900/30 px-3 py-2 text-xs text-red-300">
            {err}
          </div>
        )}
        {!selectedRunId && (
          <div className="rounded border border-gray-800 bg-gray-900 p-6 text-center text-sm text-gray-500">
            Select a run to see its resources, node timing, and metrics.
          </div>
        )}
        {selectedRunId && !detail && !err && (
          <div className="rounded border border-gray-800 bg-gray-900 p-6 text-center text-sm text-gray-500">
            Loading…
          </div>
        )}
        {detail && <RunDetailView detail={detail} />}
      </div>
    </div>
  );
}

function RunDetailView({ detail }: { detail: RunDetail }) {
  const res = detail.resources;
  const cpu = res.map((s) => s.cpu_pct);
  const mem = res.map((s) => s.mem_pct);
  const gpuIdx = Array.from(
    new Set(res.flatMap((s) => (s.gpus ?? []).map((g) => g.index))),
  ).sort((a, b) => a - b);
  const t = detail.totals;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 text-xs">
        <Stat label="firings" value={`${t.firings}`} />
        <Stat label="compute" value={`${(t.compute_ms / 1000).toFixed(1)} s`} />
        <Stat label="LLM calls" value={`${t.llm_calls}`} />
        <Stat label="tokens" value={`${t.tokens}`} />
        <Stat label="est. cost" value={`$${t.usd_cost.toFixed(4)}`} />
        <Stat
          label="transport"
          value={`${((t.transport_ms ?? 0) / 1000).toFixed(1)} s`}
        />
        <Stat
          label="total"
          value={`${((t.compute_ms + (t.transport_ms ?? 0)) / 1000).toFixed(1)} s`}
        />
        <Stat label="samples" value={`${res.length}`} />
      </div>

      {res.length > 0 ? (
        <div className="grid gap-3 lg:grid-cols-2">
          <ResourceCard
            title="CPU"
            valueLabel={avgPeak(cpu)}
            values={cpu}
            stroke="#60a5fa"
            fill="rgba(96, 165, 250, 0.15)"
          />
          <ResourceCard
            title="Memory"
            valueLabel={avgPeak(mem)}
            values={mem}
            stroke="#a78bfa"
            fill="rgba(167, 139, 250, 0.15)"
          />
          {gpuIdx.map((idx) => {
            const g = res.map(
              (s) =>
                (s.gpus ?? []).find((gg) => gg.index === idx)?.util_pct ?? 0,
            );
            return (
              <ResourceCard
                key={idx}
                title={`GPU ${idx} util`}
                valueLabel={avgPeak(g)}
                values={g}
                stroke="#34d399"
                fill="rgba(52, 211, 153, 0.15)"
              />
            );
          })}
        </div>
      ) : (
        <div className="rounded border border-gray-800 bg-gray-900 p-3 text-xs text-gray-500">
          No resource samples for this run — the sampler tees per-run
          (system.jsonl) only for runs that executed after this feature shipped.
        </div>
      )}

      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
          Node &amp; transport timing
        </h3>
        <NodeTimingTable rows={detail.node_timing} />
      </div>

      {detail.source === "eval" && detail.eval?.aggregate_metrics && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Eval metrics
            {detail.eval.total_episodes != null
              ? ` (${detail.eval.total_episodes} ep)`
              : ""}
          </h3>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(detail.eval.aggregate_metrics).map(([k, v]) => (
              <Stat
                key={k}
                label={k}
                value={typeof v === "number" ? v.toFixed(3) : String(v)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
