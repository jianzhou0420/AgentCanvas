import { useCallback, useEffect, useRef, useState } from "react";
import {
  Play,
  Square,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  ArrowUp,
  ArrowLeft as ArrowLeftIcon,
  ArrowRight as ArrowRightIcon,
  CornerDownLeft,
} from "lucide-react";
import { usePersistentState } from "../coding/usePersistentState";

// Human-performance test over env_habitat. The browser owns the control loop:
// start the env, load an episode, drive it one keypress at a time (↑ forward /
// ← turn-left / → turn-right), then Enter = STOP (with a confirm dialog) to
// score it. Metrics come from habitat's own ruler via /api/human — SR / OSR /
// NE / nDTW / SPL, identical to the coding-agent runs. Frames ride back inline
// as base64 PNG so each action is a single request.

const FRAME_PX = 512; // render size — matches the experiment RGB resolution
const N_EPISODES = 100; // rand100

type ServerState = "idle" | "starting" | "ready" | "error" | "stopped";

interface SessionView {
  index: number;
  instruction: string;
  step_count: number;
  done: boolean;
  called_stop: boolean;
  end_reason: string | null;
  metrics: Record<string, number> | null;
}
interface ServerStatus {
  state: ServerState;
  error: string | null;
  split: string;
  url: string | null;
  session: SessionView | null;
}
interface EpisodeStat {
  index: number;
  success: number | null;
  oracle_success: number | null;
  distance_to_goal: number | null;
  ndtw: number | null;
  spl: number | null;
  num_steps: number | null;
  called_stop: boolean | null;
  tested: boolean;
}
interface StatusData {
  split: string;
  episodes: EpisodeStat[];
  aggregate: Record<string, number> | null;
}

const isSuccess = (v: number | null | undefined) => (v ?? 0) > 0.5;

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function f3(v: number | null | undefined): string {
  return v == null ? "—" : v.toFixed(3);
}
function f2(v: number | null | undefined): string {
  return v == null ? "—" : v.toFixed(2);
}

export default function HumanPage() {
  const [split, setSplit] = usePersistentState("agentcanvas.human.split", "rand100");
  const [selected, setSelected] = usePersistentState<number>(
    "agentcanvas.human.selected",
    0,
  );

  const [server, setServer] = useState<ServerStatus | null>(null);
  const [status, setStatus] = useState<StatusData | null>(null);

  // Live episode state for THIS browser (frame is ephemeral — re-load to resume).
  const [frame, setFrame] = useState<string | null>(null);
  const [instruction, setInstruction] = useState<string>("");
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [stepCount, setStepCount] = useState(0);
  const [done, setDone] = useState(false);
  const [endReason, setEndReason] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);

  const [busy, setBusy] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inFlight = useRef(false); // synchronous guard against key auto-repeat

  const serverReady = server?.state === "ready";
  const canControl = serverReady && frame != null && !done && !busy && !confirmOpen;

  // ── polling ──────────────────────────────────────────────────────────
  const loadStatus = useCallback(async () => {
    try {
      const d: StatusData = await (
        await fetch(`/api/human/status?split=${encodeURIComponent(split)}`)
      ).json();
      setStatus(d);
    } catch {
      /* ignore */
    }
  }, [split]);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s: ServerStatus = await (await fetch("/api/human/server-status")).json();
        if (alive) setServer(s);
      } catch {
        /* ignore */
      }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // If the env server goes away (error / stopped — typically a backend reload
  // killed it), drop the now-stale live frame so the stage doesn't look live.
  useEffect(() => {
    if (server && server.state !== "ready" && server.state !== "starting") {
      setFrame(null);
      setActiveIndex(null);
    }
  }, [server?.state]);

  // ── server control ───────────────────────────────────────────────────
  const startServer = async () => {
    setError(null);
    try {
      await fetch("/api/human/start-server", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ split }),
      });
    } catch (e) {
      setError(String(e));
    }
  };
  const stopServer = async () => {
    await fetch("/api/human/stop-server", { method: "POST" });
    setFrame(null);
    setActiveIndex(null);
    setDone(false);
    setMetrics(null);
  };

  // ── episode control ──────────────────────────────────────────────────
  const loadEpisode = useCallback(
    async (index: number) => {
      if (!serverReady || inFlight.current) return;
      inFlight.current = true;
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(`/api/human/episode/${index}/load`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rgb_resolution: FRAME_PX }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const d = await res.json();
        setActiveIndex(index);
        setSelected(index);
        setInstruction(d.instruction || "");
        setFrame(d.frame ? `data:image/png;base64,${d.frame}` : null);
        setStepCount(0);
        setDone(false);
        setEndReason(null);
        setMetrics(null);
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [serverReady, setSelected],
  );

  const doStep = useCallback(async (action: 1 | 2 | 3) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      const res = await fetch("/api/human/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const d = await res.json();
      if (d.frame) setFrame(`data:image/png;base64,${d.frame}`);
      setStepCount(d.step_count);
      setDone(d.done);
      setEndReason(d.end_reason);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, []);

  const finishEpisode = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setConfirmOpen(false);
    try {
      const res = await fetch("/api/human/stop", { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const d = await res.json();
      setMetrics(d.metrics || {});
      setDone(true);
      setEndReason(d.end_reason);
      await loadStatus();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, [loadStatus]);

  // ── keyboard ─────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore while typing in a form field (episode-number / split inputs):
      // arrows should edit the number, Enter shouldn't open the STOP confirm.
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      // Enter opens the STOP confirm as long as an episode is live.
      if (e.key === "Enter") {
        if (serverReady && frame != null && !done && !busy) {
          e.preventDefault();
          setConfirmOpen(true);
        }
        return;
      }
      if (!canControl) return;
      if (e.key === "ArrowUp") {
        e.preventDefault();
        doStep(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        doStep(2);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        doStep(3);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canControl, serverReady, frame, done, busy, doStep]);

  // ── derived: episode grid lookup ─────────────────────────────────────
  const statByIndex = new Map<number, EpisodeStat>();
  (status?.episodes || []).forEach((e) => statByIndex.set(e.index, e));
  const testedCount = status?.episodes?.length ?? 0;
  const agg = status?.aggregate || null;

  const go = (delta: number) => {
    const next = Math.min(N_EPISODES - 1, Math.max(0, selected + delta));
    setSelected(next);
    if (serverReady) loadEpisode(next);
  };

  return (
    <div className="flex h-full bg-gray-950 text-gray-200">
      {/* ── Main stage ── */}
      <div className="flex flex-1 flex-col overflow-auto p-4">
        {/* server + episode controls */}
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="text-lg font-bold text-blue-400">Human Performance Test</span>
          <span className="ml-1 text-xs text-gray-500">
            ↑ forward · ← turn-left · → turn-right · Enter = STOP
          </span>
          <div className="flex-1" />
          <label className="text-xs text-gray-400">split</label>
          <input
            value={split}
            onChange={(e) => setSplit(e.target.value)}
            disabled={serverReady || server?.state === "starting"}
            className="w-24 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-sm disabled:opacity-50"
          />
          {!serverReady && server?.state !== "starting" ? (
            <button
              onClick={startServer}
              className="flex items-center gap-1 rounded bg-green-700 px-3 py-1 text-sm font-medium hover:bg-green-600"
            >
              <Play size={14} /> Start Session
            </button>
          ) : (
            <button
              onClick={stopServer}
              className="flex items-center gap-1 rounded bg-red-800 px-3 py-1 text-sm font-medium hover:bg-red-700"
            >
              <Square size={14} /> End Session
            </button>
          )}
          <ServerPill server={server} />
        </div>

        {(error || (server?.state === "error" && server.error)) && (
          <div className="mb-2 rounded border border-red-800 bg-red-950/60 px-3 py-1.5 text-xs text-red-300">
            {error || server?.error}
          </div>
        )}

        {/* episode nav */}
        <div className="mb-3 flex items-center gap-2">
          <button
            onClick={() => go(-1)}
            disabled={!serverReady || busy || selected <= 0}
            className="rounded border border-gray-700 bg-gray-900 p-1.5 hover:bg-gray-800 disabled:opacity-40"
            title="Previous episode"
          >
            <ChevronLeft size={16} />
          </button>
          <div className="flex items-center gap-1 text-sm">
            <span className="text-gray-400">episode</span>
            <input
              type="number"
              min={0}
              max={N_EPISODES - 1}
              value={selected}
              onChange={(e) => {
                const v = Math.min(N_EPISODES - 1, Math.max(0, Number(e.target.value) || 0));
                setSelected(v);
              }}
              className="w-16 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-center"
            />
            <span className="text-gray-500">/ {N_EPISODES - 1}</span>
          </div>
          <button
            onClick={() => go(1)}
            disabled={!serverReady || busy || selected >= N_EPISODES - 1}
            className="rounded border border-gray-700 bg-gray-900 p-1.5 hover:bg-gray-800 disabled:opacity-40"
            title="Next episode"
          >
            <ChevronRight size={16} />
          </button>
          <button
            onClick={() => loadEpisode(selected)}
            disabled={!serverReady || busy}
            className="flex items-center gap-1 rounded bg-blue-700 px-3 py-1 text-sm font-medium hover:bg-blue-600 disabled:opacity-40"
          >
            <RotateCcw size={13} />
            {activeIndex === selected && statByIndex.has(selected) ? "Re-test" : "Load"}
          </button>
          {activeIndex != null && (
            <span className="text-xs text-gray-500">
              live: ep {activeIndex} · {stepCount} steps
              {done && (
                <span className="ml-1 text-amber-400">
                  · ended{endReason ? ` (${endReason})` : ""}
                </span>
              )}
            </span>
          )}
        </div>

        {/* instruction */}
        <div className="mb-3 min-h-[3rem] rounded border border-gray-800 bg-gray-900 px-3 py-2">
          <div className="mb-0.5 text-[11px] uppercase tracking-wide text-gray-500">
            Instruction
          </div>
          <div className="text-sm text-gray-100">
            {instruction || (
              <span className="text-gray-600">
                {serverReady ? "Load an episode to begin." : "Start a session, then load an episode."}
              </span>
            )}
          </div>
        </div>

        {/* frame + controls */}
        <div className="flex flex-wrap items-start gap-4">
          <div
            className="relative shrink-0 overflow-hidden rounded border border-gray-800 bg-black"
            style={{ width: FRAME_PX, height: FRAME_PX, maxWidth: "100%" }}
          >
            {frame ? (
              <img
                src={frame}
                alt="egocentric view"
                className="h-full w-full object-contain"
                draggable={false}
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-sm text-gray-600">
                {server?.state === "starting"
                  ? "Loading habitat env… (~30s cold start)"
                  : "no frame"}
              </div>
            )}
            {busy && (
              <div className="absolute right-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs text-gray-300">
                …
              </div>
            )}
          </div>

          {/* on-screen dpad + metrics */}
          <div className="flex flex-col gap-3">
            <Dpad
              disabled={!canControl}
              onForward={() => doStep(1)}
              onLeft={() => doStep(2)}
              onRight={() => doStep(3)}
              onStop={() => frame && !done && setConfirmOpen(true)}
            />
            {/* Episode ended without a human STOP (budget exhausted): it still
                needs scoring — the normal STOP path is gated on !done. */}
            {frame && done && !metrics && (
              <div className="rounded border border-amber-800 bg-amber-950/40 p-3 text-xs">
                <div className="mb-2 text-amber-300">
                  Episode ended{endReason ? ` (${endReason})` : ""} — score it to record the result.
                </div>
                <button
                  onClick={finishEpisode}
                  disabled={busy}
                  className="w-full rounded bg-amber-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-600 disabled:opacity-40"
                >
                  Finish &amp; Score
                </button>
              </div>
            )}
            {metrics && <MetricPanel metrics={metrics} />}
          </div>
        </div>
      </div>

      {/* ── Right: episode grid + aggregate ── */}
      <div className="flex w-[300px] shrink-0 flex-col border-l border-gray-800 bg-gray-900/50">
        <div className="border-b border-gray-800 px-3 py-2">
          <div className="flex items-baseline justify-between">
            <span className="text-sm font-semibold">Episodes</span>
            <span className="text-xs text-gray-500">
              {testedCount}/{N_EPISODES} tested
            </span>
          </div>
          {agg && (
            <div className="mt-2 grid grid-cols-5 gap-1 text-center text-[10px]">
              <AggStat label="SR" value={pct(agg.success)} />
              <AggStat label="OSR" value={pct(agg.oracle_success)} />
              <AggStat label="NE" value={f2(agg.distance_to_goal)} />
              <AggStat label="nDTW" value={f3(agg.ndtw)} />
              <AggStat label="SPL" value={f3(agg.spl)} />
            </div>
          )}
        </div>
        <div className="grid grid-cols-10 gap-1 overflow-auto p-2">
          {Array.from({ length: N_EPISODES }, (_, i) => {
            const st = statByIndex.get(i);
            const tested = st?.tested;
            const ok = tested && isSuccess(st?.success);
            const cls = !tested
              ? "bg-gray-800 text-gray-500 hover:bg-gray-700"
              : ok
                ? "bg-green-700/80 text-white hover:bg-green-600"
                : "bg-red-800/80 text-white hover:bg-red-700";
            const ring =
              i === selected
                ? "ring-2 ring-blue-400"
                : i === activeIndex
                  ? "ring-2 ring-amber-400"
                  : "";
            return (
              <button
                key={i}
                onClick={() => {
                  setSelected(i);
                  if (serverReady) loadEpisode(i);
                }}
                title={
                  st
                    ? `ep ${i} · ${ok ? "success" : "fail"} · steps ${st.num_steps ?? "?"}`
                    : `ep ${i} · untested`
                }
                className={`flex h-7 items-center justify-center rounded text-[11px] font-medium ${cls} ${ring}`}
              >
                {i}
              </button>
            );
          })}
        </div>
        <div className="mt-auto border-t border-gray-800 px-3 py-2 text-[10px] text-gray-500">
          <span className="mr-2">
            <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-green-700 align-middle" />
            success
          </span>
          <span className="mr-2">
            <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-red-800 align-middle" />
            fail
          </span>
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-gray-800 align-middle" />
            untested
          </span>
        </div>
      </div>

      {/* ── STOP confirm ── */}
      {confirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-80 rounded-lg border border-gray-700 bg-gray-900 p-5 shadow-xl">
            <div className="mb-1 text-base font-semibold text-gray-100">Confirm STOP</div>
            <div className="mb-4 text-sm text-gray-400">
              Issue STOP (action 0) for episode {activeIndex} at step {stepCount}? This ends
              the episode and scores it. This cannot be undone (you can re-test afterward).
            </div>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setConfirmOpen(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-sm hover:bg-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={finishEpisode}
                className="rounded bg-red-700 px-3 py-1.5 text-sm font-medium hover:bg-red-600"
              >
                Confirm STOP
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ServerPill({ server }: { server: ServerStatus | null }) {
  const state = server?.state ?? "idle";
  const map: Record<string, string> = {
    idle: "bg-gray-700 text-gray-300",
    starting: "bg-amber-600/40 text-amber-200 animate-pulse",
    ready: "bg-green-600/40 text-green-200",
    error: "bg-red-700/50 text-red-200",
    stopped: "bg-gray-700 text-gray-400",
  };
  return (
    <span
      className={`rounded px-2 py-0.5 text-xs font-medium ${map[state] || map.idle}`}
      title={server?.error || undefined}
    >
      env: {state}
    </span>
  );
}

function AggStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-gray-800 px-1 py-1">
      <div className="text-gray-500">{label}</div>
      <div className="text-xs font-semibold text-gray-100">{value}</div>
    </div>
  );
}

function MetricPanel({ metrics }: { metrics: Record<string, number> }) {
  const ok = isSuccess(metrics.success);
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3 text-sm">
      <div className="mb-2 flex items-center gap-2">
        <span
          className={`rounded px-2 py-0.5 text-xs font-bold ${
            ok ? "bg-green-700 text-white" : "bg-red-800 text-white"
          }`}
        >
          {ok ? "SUCCESS" : "FAIL"}
        </span>
        <span className="text-xs text-gray-500">episode scored</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <Row k="SR (success)" v={ok ? "1" : "0"} />
        <Row k="OSR (oracle)" v={isSuccess(metrics.oracle_success) ? "1" : "0"} />
        <Row k="NE (dist-to-goal)" v={f2(metrics.distance_to_goal)} />
        <Row k="nDTW" v={f3(metrics.ndtw)} />
        <Row k="SPL" v={f3(metrics.spl)} />
        <Row k="path length" v={f2(metrics.path_length)} />
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <>
      <span className="text-gray-500">{k}</span>
      <span className="text-right font-medium text-gray-100">{v}</span>
    </>
  );
}

function Dpad({
  disabled,
  onForward,
  onLeft,
  onRight,
  onStop,
}: {
  disabled: boolean;
  onForward: () => void;
  onLeft: () => void;
  onRight: () => void;
  onStop: () => void;
}) {
  const btn =
    "flex items-center justify-center rounded border border-gray-700 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:hover:bg-gray-800";
  return (
    <div className="flex flex-col items-center gap-2">
      <button onClick={onForward} disabled={disabled} className={`${btn} h-10 w-10`} title="forward (↑)">
        <ArrowUp size={18} />
      </button>
      <div className="flex gap-2">
        <button onClick={onLeft} disabled={disabled} className={`${btn} h-10 w-10`} title="turn left (←)">
          <ArrowLeftIcon size={18} />
        </button>
        <button onClick={onRight} disabled={disabled} className={`${btn} h-10 w-10`} title="turn right (→)">
          <ArrowRightIcon size={18} />
        </button>
      </div>
      <button
        onClick={onStop}
        disabled={disabled}
        className="mt-1 flex items-center gap-1 rounded border border-red-800 bg-red-900/60 px-3 py-1.5 text-xs font-medium text-red-200 hover:bg-red-800/60 disabled:opacity-30"
        title="STOP (Enter)"
      >
        <CornerDownLeft size={13} /> STOP
      </button>
    </div>
  );
}
