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
  Trash2,
} from "lucide-react";
import { usePersistentState } from "../coding/usePersistentState";

// Human-performance test over env_habitat. The browser owns the control loop:
// start the env, load an episode, drive it one keypress at a time (↑/W forward,
// ←/A turn-left, →/D turn-right), then Enter = STOP (with a confirm dialog) to
// score it. Metrics come from habitat's own ruler via /api/human — SR / OSR /
// NE / nDTW / SPL, identical to the coding-agent runs. Frames ride back inline
// as base64 PNG so each action is a single request.

// Env RGB render resolution. 1024 so SAM 3 has real pixels to work with on
// distant landmarks (a bar counter down a corridor is a few dozen pixels at
// 512); the depth sensor is pushed to match by the loader.
const RENDER_PX = 1024;
const N_EPISODES = 100; // rand100
// The depth organ's pictures are sized by HEIGHT, not by container width.
// Width-sized images grew and shrank with the window, so fitting them meant
// zooming the page; height is the dimension a person compares them by.
//
// They are NOT all equal citizens. The accumulated map is the one that gets
// studied — it is where a whole walk is legible at once — while the depth frame
// and the single-frame top-down are glances that confirm the current step. So
// the two glances stack in a narrow left column and the accumulated map gets a
// tall column of its own, roughly as tall as the two of them together.
// user 2026-08-15: depth / per-frame top-down / model-view panels removed —
// the GLOBAL accumulated map is the one map shown (fields stay in the API
// type; only the display was trimmed)
const FIG_MAP = "min(60vh, 48vw)";        // the one you actually study

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
// What the depth organ made of one frame: the pictures it drew, the numbers it
// measured, the places it would offer an agent, and what SAM 3 recognised.
interface DepthAnalysis {
  depth_png: string;
  topdown_png: string;
  map_png: string;
  sam_png: string;
  map_updates: number;
  map_match: number;
  map_fixes: number;
  map_fix_m: number;
  map_drift_m: number;
  map_skipped: number;
  map_rolled_m: number;
  map_rolled_total_m: number;
  map_turn_search: boolean;
  map_trusted: boolean;
  map_no_opinion: boolean;
  map_last_score: number;
  map_walked_m: number;
  map_recall: string;
  map_semantic: {
    phrase: string;
    distance_m: number;
    bearing_deg: number;
    behind: boolean;
    cells: number;
  }[];
  annotated_png: string;
  depth_units: string;
  depth_declared: boolean;
  depth_clip_m: number | null;
  camera_height_m: number | null;
  depth_shape: number[];
  depth_min_m: number | null;
  depth_max_m: number;
  floor_below_camera_m: number;
  ahead_m: number;
  widest_bearing_deg: number;
  widest_m: number;
  sightline_m: number;
  floor_blind_m: number;
  free_pct: number;
  open_pct: number;
  occupied_pct: number;
  unknown_pct: number;
  cell_m: number;
  range_cap_m: number;
  goal_held: boolean;
  candidates: {
    n: number; kind: string; where: string; angle_deg: number;
    distance_m: number; clearance_m: number; squeeze_m: number | null;
    stride_m: number | null; boxed_in: boolean;
    verified_m?: number; visible_m?: number; confidence?: string;
    merged_deg?: number[] | null;
    env_steps: number;
  }[];
  map_model_png?: string;
  potential_regions?: { bearing_deg: number; nearest_m: number; cells: number }[];
  loop?: {
    loop_score: number; path_m: number; net_m: number;
    signals: Record<string, boolean>;
  } | null;
  landmarks: {
    phrase: string; bearing_deg: number; distance_m: number;
    score: number; pixels: number; where: string;
  }[];
  detector: string;
  surroundings: string;
  error?: string;
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
  complete?: boolean;
  archived_to?: string | null;
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
  // Clear (reset for next tester) needs a double confirmation: 0=closed,
  // 1=first prompt, 2=final prompt.
  const [clearStage, setClearStage] = useState<0 | 1 | 2>(0);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null); // green info banner
  // depth-organ inspector: drive by hand, watch what the geometry organ makes
  // of the very same frame. Analysis is on demand (or auto after each step) —
  // it never touches the episode, so it cannot perturb a scored run.
  // The drive-only tab is gone: the frame was always rendered outside the tab
  // switch, so "drive" was just the analysis panel hidden. One view now.
  const [depthData, setDepthData] = useState<DepthAnalysis | null>(null);
  // Which step the analysis describes. The candidate circles are drawn on the
  // MAIN view now, and a circle from the previous pose is worse than no circle:
  // it invites the driver to take a place the robot has already walked past.
  // So the overlay only replaces the raw frame while the two agree.
  const [depthAt, setDepthAt] = useState<number | null>(null);
  const [depthBusy, setDepthBusy] = useState(false);
  const [depthErr, setDepthErr] = useState<string | null>(null);
  const [autoDepth, setAutoDepth] = useState(true);
  const [phrasesText, setPhrasesText] = useState("pool, bar, chairs");
  const [zoomImg, setZoomImg] = useState<string | null>(null);
  // Has the first poll ever landed? Distinguishes "starting up" from "died".
  const everSeenServer = useRef(false);
  if (server !== null) everSeenServer.current = true;
  // The 300px episode grid is only needed when picking what to drive next; the
  // depth organ needs the width far more. Collapsed state is remembered, because
  // someone inspecting the map does it for a whole session, not one frame.
  const [epsOpen, setEpsOpen] = useState(
    () => localStorage.getItem("human.epsOpen") !== "0",
  );
  const toggleEps = useCallback(() => {
    setEpsOpen((v) => {
      localStorage.setItem("human.epsOpen", v ? "0" : "1");
      return !v;
    });
  }, []);

  const inFlight = useRef(false); // synchronous guard against key auto-repeat

  const serverReady = server?.state === "ready";
  // `server` is null only while the very first poll is in flight; after that a
  // null means the poll itself failed, i.e. the BACKEND is gone. Vite proxies
  // /api to it, so a dead backend surfaces in the browser as a plain 500 on
  // every call and the page looks alive but inert — which is indistinguishable
  // from "the env is loading" unless the page says which layer is down.
  const backendDown = server === null;
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
  const runDepthAnalysisWith = useCallback(
    async (phrases: string[], opts?: { resetMap?: boolean }) => {
      setDepthBusy(true);
      setDepthErr(null);
      try {
        const res = await fetch("/api/human/depth-analysis", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            phrases,
            ...(opts?.resetMap ? { reset_map: true } : {}),
          }),
        });
        if (!res.ok) {
          setDepthErr(`${res.status}: ${(await res.text()).slice(0, 300)}`);
          return false;
        }
        setDepthData(await res.json());
        setDepthAt(stepCount);
        return true;
      } catch (e) {
        setDepthErr(String(e instanceof Error ? e.message : e));
        return false;
      } finally {
        setDepthBusy(false);
      }
    },
    [stepCount],
  ); // stepCount is captured for depthAt

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
          body: JSON.stringify({ rgb_resolution: RENDER_PX, depth_resolution: 512 }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const d = await res.json();
        setActiveIndex(index);
        setSelected(index);
        setInstruction(d.instruction || "");
        // refresh the SAM phrase box from THIS episode's instruction — the
        // old hardcoded default meant every episode was probed for EP0's
        // "pool, bar, chairs" until hand-edited (stale-keyword bug)
        if (Array.isArray(d.suggested_phrases) && d.suggested_phrases.length)
          setPhrasesText(d.suggested_phrases.join(", "));
        setFrame(d.frame ? `data:image/png;base64,${d.frame}` : null);
        setStepCount(0);
        setDone(false);
        setEndReason(null);
        setMetrics(null);
        // auto-analyse with THIS episode's phrases (state above is async —
        // pass them explicitly): the backend opens every fresh map with the
        // zero-step ±60° sweep, so the heading-up accumulated panel shows
        // the surroundings immediately, before a single step
        void (async () => {
          const ph = Array.isArray(d.suggested_phrases)
            ? d.suggested_phrases
            : [];
          const ok = await runDepthAnalysisWith(ph, { resetMap: true });
          // the env session can still be settling right after load — one
          // quiet retry covers the race that left the map panel empty
          // until the first manual turn
          if (!ok) {
            await new Promise((r) => setTimeout(r, 1500));
            await runDepthAnalysisWith(ph);
          }
        })();
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [serverReady, setSelected, runDepthAnalysisWith],
  );

  const runDepthAnalysis = useCallback(async () => {
    const phrases = phrasesText
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
    await runDepthAnalysisWith(phrases);
  }, [phrasesText, runDepthAnalysisWith]);

  // re-analyse whenever the robot has moved, so the readout always describes
  // the frame on the left rather than a stale one
  useEffect(() => {
    if (autoDepth && frame && !busy) void runDepthAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoDepth, stepCount, frame]);

  // Take one of the numbered places the depth organ is offering. Same response
  // shape as a step, so the stage updates identically — the difference is that
  // one call covers several metres along a line that was measured clear.
  // The backend recomputes the candidates from the CURRENT frame rather than
  // trusting whatever the inspector last drew: moving invalidates them.
  const doGoto = useCallback(async (place: number) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      const res = await fetch("/api/human/goto", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ place }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const d = await res.json();
      if (d.frame) setFrame(`data:image/png;base64,${d.frame}`);
      setStepCount(d.step_count);
      setDone(d.done);
      setEndReason(d.end_reason);
      setNotice(`goto #${place} → ${d.went_to}（走了 ${d.env_steps_used} 步）`);
      setError(null);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, []);

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

  // Reset the working set for the next tester. The backend archives first
  // (unless this exact data is already saved), so nothing is lost.
  const clearData = useCallback(async () => {
    setClearStage(0);
    setError(null);
    try {
      const res = await fetch("/api/human/clear", { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const d = await res.json();
      setFrame(null);
      setActiveIndex(null);
      setDone(false);
      setMetrics(null);
      setInstruction("");
      setStepCount(0);
      setStatus(d);
      setNotice(
        d.archived_to
          ? `Saved to archive/${d.archived_to} · grid cleared for the next tester.`
          : "Grid cleared for the next tester.",
      );
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  // ── keyboard ─────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore while typing in a form field (episode-number / split inputs):
      // arrows should edit the number, Enter shouldn't open the STOP confirm.
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      // With the confirm dialog open, Enter confirms (STOP) and Escape cancels.
      if (confirmOpen) {
        if (e.key === "Enter") {
          e.preventDefault();
          finishEpisode();
        } else if (e.key === "Escape") {
          e.preventDefault();
          setConfirmOpen(false);
        }
        return;
      }
      // Enter opens the STOP confirm as long as an episode is live.
      if (e.key === "Enter") {
        if (serverReady && frame != null && !done && !busy) {
          e.preventDefault();
          setConfirmOpen(true);
        }
        return;
      }
      if (!canControl) return;
      // 1/2/3 take the numbered place the depth organ is offering — the same
      // verb the agent has. Driving by hand with WASD and driving with goto are
      // different questions: WASD asks "is this route walkable", goto asks "is
      // the proposer offering the place a person would take".
      if (e.key === "1" || e.key === "2" || e.key === "3") {
        e.preventDefault();
        doGoto(Number(e.key));
        return;
      }
      // WASD alongside the arrows — one hand on the keys, one on the mouse for
      // the organ panel. Case-insensitive so Caps Lock does not disarm driving.
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      if (k === "ArrowUp" || k === "w") {
        e.preventDefault();
        doStep(1);
      } else if (k === "ArrowLeft" || k === "a") {
        e.preventDefault();
        doStep(2);
      } else if (k === "ArrowRight" || k === "d") {
        e.preventDefault();
        doStep(3);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canControl, serverReady, frame, done, busy, doStep, doGoto, confirmOpen, finishEpisode]);

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
      {/* click any organ image to inspect it full size */}
      {zoomImg && (
        <div
          onClick={() => setZoomImg(null)}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/85 p-6"
        >
          <img src={zoomImg} alt="zoom" className="max-h-full max-w-full object-contain" />
        </div>
      )}
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

        {backendDown && everSeenServer.current && (
          <div className="mb-2 rounded border border-amber-700 bg-amber-950/50 px-3 py-1.5 text-xs text-amber-200">
            后端 (:8010) 没有响应 —— 页面本身是活的，是 <code>/api</code> 代理的目标挂了。
            所有按钮都会失败，直到它回来。启动：
            <code className="ml-1 rounded bg-black/40 px-1">
              cd agentcanvas/backend &amp;&amp; setsid nohup python -m uvicorn app.main:app --port 8010 &amp;
            </code>
            <span className="ml-1 opacity-70">
              （不要加 --reload：它会杀掉自己 spawn 的 habitat 子进程）
            </span>
          </div>
        )}
        {!backendDown && server?.state !== "ready" && server?.state !== "starting" && (
          <div className="mb-2 rounded border border-sky-800 bg-sky-950/40 px-3 py-1.5 text-xs text-sky-300">
            后端在，但 habitat env 还没起（<code>{server?.state}</code>）—— 点「启动服务器」。
          </div>
        )}
        {(error || (server?.state === "error" && server.error)) && (
          <div className="mb-2 rounded border border-red-800 bg-red-950/60 px-3 py-1.5 text-xs text-red-300">
            {error || server?.error}
          </div>
        )}
        {(notice || (status?.complete && status?.archived_to)) && (
          <div className="mb-2 flex items-center justify-between gap-2 rounded border border-green-800 bg-green-950/50 px-3 py-1.5 text-xs text-green-300">
            <span>
              {notice ??
                `✓ All ${N_EPISODES} episodes tested — auto-saved to archive/${status?.archived_to}. Clear to reset for the next tester.`}
            </span>
            {notice && (
              <button onClick={() => setNotice(null)} className="text-green-500 hover:text-green-300">
                ✕
              </button>
            )}
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

        <div className="mb-3 border-b border-gray-800 px-1 pb-1.5 text-sm text-cyan-300">
          analysis
        </div>

        {/* frame + controls */}
        <div className="flex flex-wrap items-start justify-center gap-4">
          <div
            className="relative shrink-0 overflow-hidden rounded border border-gray-800 bg-black"
            // Big, square view that fills the left panel: sized by viewport
            // height (so it stays fully visible), capped by the column width.
            // compact (user 2026-08-12 night): the photo cedes room to the map
            style={{ width: "min(46vh, 100%)", aspectRatio: "1 / 1", maxWidth: "100%" }}
          >
            {frame ? (
              <img
                src={
                  depthData?.annotated_png && depthAt === stepCount
                    ? `data:image/png;base64,${depthData.annotated_png}`
                    : frame
                }
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

          {/* the organ's readout, beside the very frame it was computed from */}
          {(
            <div className="min-w-[340px] flex-1 space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => runDepthAnalysis()}
                  disabled={depthBusy || !frame}
                  className="rounded bg-cyan-800 px-3 py-1 text-white hover:bg-cyan-700 disabled:opacity-40"
                >
                  {depthBusy ? "分析中…" : "分析当前帧"}
                </button>
                <label className="flex items-center gap-1 text-gray-400">
                  <input
                    type="checkbox"
                    checked={autoDepth}
                    onChange={(e) => setAutoDepth(e.target.checked)}
                  />
                  每步自动分析
                </label>
              </div>
              <input
                value={phrasesText}
                onChange={(e) => setPhrasesText(e.target.value)}
                placeholder="SAM3 要找的地标，逗号分隔（留空则不跑检测）"
                className="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1 text-gray-200"
              />
              {depthErr && (
                <div className="rounded border border-red-800 bg-red-950/40 p-2 text-red-300">
                  {depthErr}
                </div>
              )}
              {depthData && !depthData.error && (
                <>
                  {/* Two columns, both height-driven so nothing resizes with the
                      window: the two per-frame glances stack on the left, and
                      the accumulated map takes a tall column of its own on the
                      right. The row scrolls sideways rather than reflowing, so
                      the maps never move on you. */}
                  {/* user 2026-08-15: ONE map only — the anchor-fixed
                      GLOBAL accumulated view. Depth / per-frame top-down /
                      model-view duplicates removed from display. */}
                  <div className="flex items-start gap-2 overflow-x-auto pb-1">
                    {depthData.map_png && (
                      <figure className="shrink-0">
                        <img
                          src={`data:image/png;base64,${depthData.map_png}`}
                          alt="accumulated map"
                          onClick={() =>
                            setZoomImg(`data:image/png;base64,${depthData.map_png}`)
                          }
                          style={{ height: FIG_MAP }}
                          className="w-auto max-w-none cursor-zoom-in rounded border border-indigo-900/70 bg-black"
                        />
                        <figcaption
                          className="mt-0.5 max-w-[420px] text-[10px] leading-tight text-gray-500"
                          title="2.5D：深度压成占据+语义俯视层；楼梯/多层/悬空障碍不在此图的表达范围"
                        >
                          2.5D accumulated GLOBAL map（<b>世界固定</b>：墙/轨迹/编号圆不动，
                          黄箭头是你、随真实朝向转）· 紫线=轨迹（近段更亮）· 深紫=身体到过的格 ·
                          彩色=SAM 认出的东西 · 琥珀=瞥见未验证 · 已融合{" "}
                          {depthData.map_updates} 帧 · 走了 {depthData.map_walked_m}m
                        </figcaption>
                      </figure>
                    )}
                  </div>
                  {(depthData.potential_regions?.length || depthData.loop) && (
                    <div className="rounded border border-amber-900/60 bg-amber-950/20 p-2 text-[11px]">
                      {depthData.potential_regions?.map((r, i) => (
                        <div key={i} className="text-amber-300">
                          ◮ 瞥见未验证的开阔地：
                          {Math.abs(r.bearing_deg) < 12
                            ? "正前方"
                            : `${r.bearing_deg > 0 ? "左" : "右"} ${Math.abs(r.bearing_deg)}°`}
                          ，最近约 {r.nearest_m}m（{r.cells} 格）— 中间是 UNKNOWN，不会盲走
                        </div>
                      ))}
                      {depthData.loop && (
                        <div className="text-red-300">
                          ⟲ LOOP WARNING（{depthData.loop.loop_score}）：走了{" "}
                          {depthData.loop.path_m}m 净位移只有 {depthData.loop.net_m}m —
                          轨迹显示这片区域走过了（
                          {Object.keys(depthData.loop.signals).join("·")}）
                        </div>
                      )}
                    </div>
                  )}
                  {depthData.map_png && (
                    <div className="flex flex-col gap-2 lg:flex-row lg:items-start">
                      <div className="min-w-0 flex-1 space-y-2">
                        {/* what the drift study changed, visible while driving */}
                        <div className="rounded border border-indigo-900/60 bg-indigo-950/20 p-2">
                          <div className="mb-1 text-indigo-300">位姿与配准</div>
                          <div className="text-gray-300">
                            {depthData.map_no_opinion ? (
                              <span className="text-gray-500">
                                本帧无法配准（视野结构太少）· 上次 {depthData.map_last_score}
                              </span>
                            ) : (
                              <>配准 {depthData.map_match}</>
                            )}
                            <span className="text-gray-600">
                              {" "}
                              · {depthData.map_turn_search ? "搜索旋转" : "不搜旋转"}
                            </span>
                          </div>
                          <div className="text-gray-300">
                            {depthData.map_fixes > 0
                              ? `已修正 ${depthData.map_fixes} 次（上次 ${depthData.map_fix_m}m）`
                              : "尚未需要修正"}
                          </div>
                          {depthData.map_rolled_m > 0 ? (
                            <div className="mt-1 rounded bg-amber-950/50 px-1 py-0.5 text-amber-300">
                              这一步撞住了：画面没变，{depthData.map_rolled_m}m
                              已从里程里扣回
                            </div>
                          ) : (
                            <div className="text-gray-600">这一步画面有变化，正常前进</div>
                          )}
                          {depthData.map_rolled_total_m > 0 && (
                            <div className="text-[10px] text-gray-500">
                              本集累计扣回 {depthData.map_rolled_total_m}m（命令走了、身体没走）
                            </div>
                          )}
                          {!depthData.map_trusted && (
                            <div className="mt-1 text-[10px] text-amber-400">
                              配准质量已掉线 —— 地图暂停报米数，只报方位
                            </div>
                          )}
                          {depthData.map_skipped > 0 && (
                            <div className="text-[10px] text-gray-500">
                              {depthData.map_skipped} 步没分析（位姿照推，只是少了证据）
                            </div>
                          )}
                        </div>
                      </div>
                      {depthData.map_semantic?.length > 0 && (
                        <div className="min-w-0 flex-1">
                          <div className="rounded border border-fuchsia-900/60 bg-fuchsia-950/20 p-2">
                            <div className="mb-1 text-fuchsia-300">
                              语义地图 · {depthData.map_semantic.length}{" "}
                              个已落图的地标（看不见了也还记得）
                            </div>
                            {depthData.map_semantic.map((s) => (
                              <div key={s.phrase} className="text-gray-300">
                                {s.phrase} · {s.distance_m}m ·{" "}
                                {s.behind
                                  ? "在你身后"
                                  : `${s.bearing_deg > 0 ? "左" : "右"}${Math.abs(s.bearing_deg)}°`}{" "}
                                <span className="text-gray-600">({s.cells} 格)</span>
                              </div>
                            ))}
                            {depthData.map_recall && (
                              <div className="mt-1 text-[10px] text-gray-500">
                                给模型的句子：{depthData.map_recall}
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  <div className="rounded border border-gray-800 bg-gray-900/60 p-2">
                    <div className="text-gray-300">{depthData.surroundings}</div>
                    <div className="mt-1 text-gray-500">
                      能走 {depthData.ahead_m}m · 能看 {depthData.sightline_m}m ·
                      最开阔{" "}
                      {depthData.widest_bearing_deg > 0 ? "左" : "右"}
                      {Math.abs(depthData.widest_bearing_deg)}° @ {depthData.widest_m}m ·
                      free {depthData.free_pct}% open {depthData.open_pct}% occ{" "}
                      {depthData.occupied_pct}% unknown {depthData.unknown_pct}% ·
                      地面在相机下方 {depthData.floor_below_camera_m}m
                    </div>
                    {/* 相机装配。视高和深度上限都不是常识，是配置——写出来，
                        免得再有人把 estimate_floor 的兜底常量当成量出来的高度。 */}
                    <div className="mt-1 text-gray-600">
                      视高{" "}
                      {depthData.camera_height_m === null
                        ? "?"
                        : `${depthData.camera_height_m}m`}
                      （habitat 默认 1.25）· 深度上限{" "}
                      {depthData.depth_clip_m === null
                        ? "?"
                        : `${depthData.depth_clip_m}m`}
                      （默认 10）· 单位 {depthData.depth_units}
                      {!depthData.depth_declared && (
                        <span className="ml-1 text-amber-500">
                          —— env 没说单位，是猜的
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="rounded border border-cyan-900/60 bg-cyan-950/20 p-2">
                    <div className="mb-1 text-cyan-300">
                      dwp 候选（agent 会看到的选项）· {depthData.candidates.length} 个
                      {depthData.goal_held && (
                        <span className="ml-1 text-emerald-400">
                          · 1 是已认定的目标，走近它距离会往下掉
                        </span>
                      )}
                      <span className="ml-1 text-gray-500">
                        —— 点一下，或按 1/2/3，直接 goto
                      </span>
                    </div>
                    {depthData.candidates.length === 0 && (
                      <div className="text-gray-500">
                        没有候选 —— 前方被挡或太窄，转身再看
                      </div>
                    )}
                    {depthData.candidates.map((c) => (
                      <button
                        key={c.n}
                        onClick={() => doGoto(c.n)}
                        disabled={!canControl}
                        title="走到这个位置（键盘 1/2/3 同）"
                        className="block w-full rounded px-1 py-0.5 text-left text-gray-300 hover:bg-cyan-900/30 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <span className="text-cyan-400">{c.n}</span>{" "}
                        <span className="text-gray-500">[{c.kind}]</span> {c.where}
                        <span className="text-gray-600">
                          {" "}
                          · 走 {c.stride_m ?? c.distance_m}m/{c.env_steps} 步
                          {c.verified_m != null ? ` · 已验证 ${c.verified_m}m` : ""}
                          {c.visible_m != null ? ` · 视线 ${c.visible_m}m` : ""}
                          {c.confidence ? ` · ${c.confidence}` : ""}
                          {c.squeeze_m != null ? ` · 最窄 ${c.squeeze_m}m` : ""}
                          {c.merged_deg?.length
                            ? ` · 并入了 ${c.merged_deg.join("°, ")}° 方向`
                            : ""}
                        </span>
                      </button>
                    ))}
                  </div>
                  <div className="rounded border border-emerald-900/60 bg-emerald-950/20 p-2">
                    <div className="mb-1 text-emerald-300">
                      SAM 3 · {depthData.detector}
                    </div>
                    {depthData.sam_png && (
                      <img
                        src={`data:image/png;base64,${depthData.sam_png}`}
                        alt="sam3 masks"
                        onClick={() =>
                          setZoomImg(`data:image/png;base64,${depthData.sam_png}`)
                        }
                        className="mb-1 cursor-zoom-in rounded border border-emerald-900/60"
                        style={{ maxHeight: "24vh", width: "auto", maxWidth: "100%" }}
                      />
                    )}
                    {depthData.landmarks.length === 0 ? (
                      <div className="text-gray-500">
                        {depthData.detector === "off"
                          ? "未启用（上面填地标词再分析）"
                          : "这一帧没认出任何指定地标"}
                      </div>
                    ) : (
                      depthData.landmarks.map((l, i) => (
                        <div key={i} className="text-gray-300">
                          {l.where}
                          <span className="text-gray-600">
                            {" "}
                            · score {l.score} · {l.pixels}px
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                  {/* the annotated view is the MAIN image now, not a thumbnail */}
                </>
              )}
            </div>
          )}

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

      {/* ── Right: episode grid + aggregate (collapsible) ── */}
      {!epsOpen && (
        <button
          onClick={toggleEps}
          title="展开 Episodes"
          className="flex w-8 shrink-0 flex-col items-center gap-2 border-l border-gray-800 bg-gray-900/50 py-2 text-gray-500 hover:bg-gray-800/60 hover:text-gray-300"
        >
          <ChevronLeft size={14} />
          <span
            className="text-[10px] tracking-wider"
            style={{ writingMode: "vertical-rl" }}
          >
            Episodes {testedCount}/{N_EPISODES}
          </span>
        </button>
      )}
      <div
        className={`${epsOpen ? "flex" : "hidden"} w-[300px] shrink-0 flex-col border-l border-gray-800 bg-gray-900/50`}
      >
        <div className="border-b border-gray-800 px-3 py-2">
          <div className="flex items-center justify-between">
            <button
              onClick={toggleEps}
              title="收起 Episodes，把版面让给 analysis"
              className="flex items-center gap-1 text-sm font-semibold text-gray-200 hover:text-white"
            >
              <ChevronRight size={14} /> Episodes
            </button>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">
                {testedCount}/{N_EPISODES} tested
              </span>
              <button
                onClick={() => setClearStage(1)}
                disabled={testedCount === 0}
                className="flex items-center gap-1 rounded border border-red-800 bg-red-950/40 px-2 py-0.5 text-xs text-red-300 hover:bg-red-900/50 disabled:opacity-40"
                title="Clear all data and reset the grid for the next tester"
              >
                <Trash2 size={12} /> Clear
              </button>
            </div>
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

      {/* ── Clear: double confirmation (reset for the next tester) ── */}
      {clearStage > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="w-96 rounded-lg border border-gray-700 bg-gray-900 p-5 shadow-xl">
            {clearStage === 1 ? (
              <>
                <div className="mb-1 text-base font-semibold text-gray-100">
                  Clear all data for the next tester?
                </div>
                <div className="mb-4 text-sm text-gray-400">
                  This resets the grid so a new person can test the {N_EPISODES} episodes
                  from scratch. The current run ({testedCount}/{N_EPISODES} tested) is saved
                  to an archive first — nothing is lost.
                </div>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setClearStage(0)}
                    className="rounded border border-gray-700 px-3 py-1.5 text-sm hover:bg-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => setClearStage(2)}
                    className="rounded bg-red-800 px-3 py-1.5 text-sm font-medium hover:bg-red-700"
                  >
                    Continue
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="mb-1 text-base font-semibold text-red-300">
                  Final confirmation
                </div>
                <div className="mb-4 text-sm text-gray-400">
                  Permanently wipe the live {testedCount}-episode grid? It has been archived
                  (recoverable there), but the working grid cannot be un-cleared.
                </div>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setClearStage(0)}
                    className="rounded border border-gray-700 px-3 py-1.5 text-sm hover:bg-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={clearData}
                    className="rounded bg-red-700 px-3 py-1.5 text-sm font-medium hover:bg-red-600"
                  >
                    Yes, clear everything
                  </button>
                </div>
              </>
            )}
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
