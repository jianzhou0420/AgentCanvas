import { useEffect, useRef, useState } from "react";
import { BarChart3, Bot, Download, Play, Square } from "lucide-react";
import clsx from "clsx";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import { usePersistentState } from "./usePersistentState";

// Coding-Agent Monitor — control panel + live text/image logs for coding-agent
// runs (vanilla coding agent driving env_habitat through the MCP bridge;
// artifacts stay under the legacy outputs/beta-* roots).
// v1: single worker, one run at a time. Live data is 1 Hz polling against
// /api/coding-agent (the trajectory JSONL is flushed per event backend-side).

const POLL_MS = 1000;

// board model keys per harness (mirrors coding-agent/cells.py: MODELS + the
// BOARD / WP_BOARD / QWEN_API_BOARD / LOCAL_BOARD pairings; the driver
// resolves a key to its concrete slug + per-model knobs via cells.py).
// "" = the harness's own default model (sdk account / codex CLI); mini-swe
// has no default — litellm needs an explicit model.
const MODEL_OPTIONS: Record<string, { value: string; label: string }[]> = {
  "claude-sdk": [
    { value: "", label: "(account default)" },
    { value: "sonnet-5", label: "sonnet-5" },
    { value: "opus-4.8", label: "opus-4.8" },
    { value: "opus-5", label: "opus-5" },
    { value: "fable-5", label: "fable-5" },
  ],
  codex: [
    { value: "", label: "(CLI default)" },
    { value: "gpt-5.5", label: "gpt-5.5" },
    { value: "gpt-5.6", label: "gpt-5.6" },
  ],
  "mini-swe": [
    { value: "sonnet-5", label: "sonnet-5" },
    { value: "opus-4.8", label: "opus-4.8" },
    { value: "fable-5", label: "fable-5" },
    { value: "gpt-5.5", label: "gpt-5.5" },
    { value: "gpt-5.6", label: "gpt-5.6" },
    { value: "qwen3.5-plus", label: "qwen3.5-plus" },
    { value: "qwen3.6-plus", label: "qwen3.6-plus" },
    { value: "qwen3.7-plus", label: "qwen3.7-plus" },
    { value: "qwen3.8-plus", label: "qwen3.8-plus" },
    { value: "qwen3.5-4b", label: "qwen3.5-4b (local)" },
    { value: "qwen3.5-9b", label: "qwen3.5-9b (local)" },
  ],
};

interface EpisodeSummary {
  index: number;
  success: number | null;
  spl: number | null;
  distance_to_goal: number | null;
  env_steps: number | null;
  called_stop: boolean | null;
  error: string | null;
}

interface RunStatus {
  state: string;
  run_name: string | null;
  error: string | null;
  config: {
    episodes?: string;
    split?: string;
    max_turns?: number;
    model?: string | null;
    harness?: string;
    condition?: string;
    tier?: string;
    extra?: Record<string, string>;
  };
  active_episode: number | null;
  started_episodes: number[];
  aggregate: Record<string, number> | null;
  episodes: EpisodeSummary[];
}

interface LogLine {
  t: number;
  kind: string;
  [key: string]: unknown;
}

interface RunInfo {
  name: string;
  mtime: number;
  episodes: number[];
  success?: number | null;
  episode_count?: number | null;
  model?: string | null;
  skill?: string | null;
}

function lineText(line: LogLine): { icon: string; text: string; dim: boolean } {
  switch (line.kind) {
    case "episode_meta":
      return { icon: "📍", text: `episode ${line.index} · ${line.instruction}`, dim: false };
    case "system_init":
      return { icon: "⚙", text: `session up · model=${line.model}`, dim: true };
    case "bridge_status":
      return { icon: "🔌", text: `bridge ${line.status}`, dim: true };
    case "thinking":
      // With thinking display "summarized" the block carries a readable
      // reasoning summary; older runs have empty signature-only blocks.
      return {
        icon: "🤔",
        text: line.text ? String(line.text) : `thinking… (${line.chars} chars)`,
        dim: true,
      };
    case "assistant_text":
      return { icon: "💬", text: String(line.text ?? ""), dim: false };
    case "tool_use": {
      const name = String(line.name ?? "").split("__").pop();
      return {
        icon: line.tier === "planner" ? "🧭" : "🔧",
        text: `${name} ${JSON.stringify(line.input)}`,
        dim: false,
      };
    }
    case "tool_result": {
      const texts = (line.texts as string[] | undefined) ?? [];
      return { icon: "↩", text: texts.join(" ").slice(0, 300), dim: true };
    }
    // mini-swe-agent event kinds (mini harness runs)
    case "user_text":
      return { icon: "👤", text: String(line.text ?? ""), dim: true };
    case "exit":
      return {
        icon: "🚪",
        text: `exit · ${String(line.exit_status ?? "?")} ${String(line.content ?? "").slice(0, 200)}`,
        dim: false,
      };
    case "driver_error":
      return { icon: "⚠", text: String(line.error ?? ""), dim: false };
    case "episode_metrics": {
      // The scored outcome of the episode — the one line a reader looks for,
      // so it renders as text rather than falling through to raw JSON.
      const m = (line.metrics ?? {}) as Record<string, number>;
      const n = (v: number | undefined, d = 2) =>
        typeof v === "number" ? v.toFixed(d) : "—";
      const verdict = m.success ? "SUCCESS" : "FAIL";
      const salvaged = line.salvaged ? " · metrics salvaged post-hoc" : "";
      return {
        icon: m.success ? "✅" : "❌",
        text:
          `${verdict} · NE ${n(m.distance_to_goal)}m · SPL ${n(m.spl)} · ` +
          `nDTW ${n(m.ndtw)} · OSR ${n(m.oracle_success, 0)} · ` +
          `steps ${n(m.steps_taken, 0)}${salvaged}`,
        dim: false,
      };
    }
    // ── eharness organ events (state_write / guard / verdict / compact /
    //    receipt / recall) — the loop's narrative, one readable line each ──
    case "state_write": {
      const u = (line.update ?? {}) as Record<string, unknown>;
      // Defensive: a driver that writes this as a string must not blank the
      // whole page on .join().
      const acc = Array.isArray(line.accepted) ? (line.accepted as string[]) : [];
      const bits = [
        u.advance_subgoal ? "advance✓" : null,
        u.landmark ? `landmark:'${String(u.landmark)}'` : null,
        u.dead_end ? `dead-end:'${String(u.dead_end)}'` : null,
        u.progress_note ? `note:'${String(u.progress_note).slice(0, 70)}'` : null,
      ].filter(Boolean);
      return {
        icon: "📝",
        text: `state · ${bits.join(" · ") || JSON.stringify(u)}` +
              (acc.length ? ` → ${acc.join("; ")}` : ""),
        dim: false,
      };
    }
    case "guard":
      return {
        icon: "🛑",
        text: `guard${line.escalate ? " · ESCALATE" : ""} · ${String(line.notice ?? "").slice(0, 220)}`,
        dim: false,
      };
    case "verdict": {
      const v = line.supported === true ? "PASS"
        : line.supported === false ? "VETO" : "no-verdict";
      return {
        icon: "⚖",
        text: `verdict[${String(line.gate ?? "?")}] ${v}` +
              (line.claim ? ` · '${String(line.claim).slice(0, 80)}'` : "") +
              ` · ${String(line.reason ?? "").slice(0, 160)}`,
        dim: false,
      };
    }
    case "compact":
      return {
        icon: "🗜",
        text: `compact · ~${line.est_tokens} tok → dropped ${line.dropped_msgs} msgs · re-attached ${line.reattached} keyframes`,
        dim: false,
      };
    case "receipt":
      return {
        icon: "📦",
        text: `receipt · ${String(line.claim ?? "?")} · '${String(line.subgoal ?? "").slice(0, 90)}' · ` +
              `steps ${line.steps_used} · ${String(line.end_reason ?? "")}` +
              (line.not_done ? ` · not done: ${String(line.not_done).slice(0, 90)}` : ""),
        dim: false,
      };
    case "recall":
      return { icon: "🖼", text: `recall '${String(line.query ?? "")}' → ${line.hits} hit(s)`, dim: true };
    case "crisis":
      return {
        icon: "🧩",
        text: `crisis resolved · frames ${line.start}→${line.end} collapsed · ${String(line.lesson ?? "").slice(0, 160)}`,
        dim: false,
      };
    default:
      return { icon: "·", text: JSON.stringify(line), dim: true };
  }
}

// eharness organ readout (state.json + heartbeat.json + keyframes + receipts)
interface HarnessData {
  state: {
    sub_instructions?: string[];
    cursor?: number;
    current_place?: string;
    landmarks?: Record<string, unknown>;
    negative_facts?: { fact: string; verified?: boolean }[];
    checkpoint?: string;
    delegation?: { subgoal: string; budget: number; used: number } | null;
    last_action?: string;
    expectation?: string;
    _rendered?: string;
  } | null;
  heartbeat: {
    status?: string; subgoal?: string; steps_used?: number;
    steps_budget?: number; tool_calls?: number; guard_trips?: number;
    last_note?: string;
  } | null;
  // depth-waypoint runs: what the geometry organ measured and believed. The
  // model never sees this — it is here so a human can watch the map the
  // harness is navigating by.
  depth: {
    obs?: number; steps_taken?: number; gotos?: number; depth_units?: string;
    floor_below_camera_m?: number; range_cap_m?: number; cell_m?: number;
    ahead_m?: number; widest_bearing_deg?: number; widest_m?: number;
    free_pct?: number; occupied_pct?: number; unknown_pct?: number;
    detector?: string;
    candidates?: {
      n: number; kind: string; angle_deg: number; distance_m: number;
      clearance_m: number; squeeze_m?: number; env_steps: number; where: string;
    }[];
    landmarks?: { phrase: string; bearing_deg: number; distance_m: number; score: number }[];
    landmark_ledger?: Record<string, string>;
  } | null;
  keyframes: { idx: number; step: number; events: string[]; png: string }[];
  segments: { seg: number; label: string; route: string; motion: string; png: string }[];
  receipts: {
    subgoal: string; claim: string; verdict?: string;
    steps_used?: number; not_done?: string;
  }[];
}

export default function CodingAgentPage() {
  // control panel form (persisted: survive a refresh with the same inputs)
  const [episodes, setEpisodes] = usePersistentState("agentcanvas.coding.episodes", "0-9");
  const [split, setSplit] = usePersistentState("agentcanvas.coding.split", "rand100");
  // .v2 keys: defaults moved to the std-v2 board values (max_turns 200,
  // model fable-5) — bumped so stale persisted values don't shadow them
  const [maxTurns, setMaxTurns] = usePersistentState("agentcanvas.coding.maxTurns.v2", 200);
  const [modelRaw, setModel] = usePersistentState("agentcanvas.coding.model.v2", "fable-5");
  const [conditionRaw, setCondition] = usePersistentState("agentcanvas.coding.condition", "bare");
  // "ui" was retired; a stale persisted value falls back to bare
  const condition = ["bare", "wp", "hybrid"].includes(conditionRaw) ? conditionRaw : "bare";
  const [tier, setTier] = usePersistentState("agentcanvas.coding.tier", "default");
  // free-form harness knobs, "k=v k=v" (forwarded as --set pairs, override tier's)
  const [extraText, setExtraText] = usePersistentState("agentcanvas.coding.extra", "");
  const [startError, setStartError] = useState<string | null>(null);

  // live state
  const [status, setStatus] = useState<RunStatus | null>(null);
  // which episode's log/frames are shown (null = follow active). Persisted so a
  // refresh keeps you on the same episode; in live mode the poll loop below
  // resets it to null on a run change, so a stale index can't stick.
  const [viewEpisode, setViewEpisode] = usePersistentState<number | null>(
    "agentcanvas.coding.viewEpisode",
    null,
  );
  const [lines, setLines] = useState<LogLine[]>([]);
  const [frames, setFrames] = useState<string[]>([]);
  const [zoomFrame, setZoomFrame] = useState<string | null>(null); // lightbox overlay
  const [harnessData, setHarnessData] = useState<HarnessData | null>(null);
  // the top-down map is only written by depth-waypoint runs; one 404 retires
  // the panel for this episode rather than flashing a broken image forever
  const [topdownDead, setTopdownDead] = useState(false);
  const [amapDead, setAmapDead] = useState(false);
  const [snap, setSnap] = useState<{
    obs_id: number; env_step: number; map_version: number;
    rgb_file: string; topdown_file: string; accumulated_map_file: string;
    model_map_file: string; consistent: boolean;
    // §14.14 snapshot identity — absent on runs recorded before the field
    // existed, so every reader must tolerate undefined
    identity?: {
      run_id?: string; episode?: string; executor?: string;
      action_id?: string; sensor_frame?: number;
    } | null;
    missing?: string[];
  } | null>(null);
  const [shownRunEp, setShownRunEp] = useState("");
  // organ panel is tall — collapsible, and its body scrolls internally so the
  // log below always keeps its space
  const [showHarness, setShowHarness] = usePersistentState(
    "agentcanvas.coding.showHarness",
    true,
  );

  // log browser (any run under outputs/beta-coding-agent/, CLI-launched included)
  const [mode, setMode] = usePersistentState<"live" | "browse">(
    "agentcanvas.coding.mode",
    "live",
  );
  // which harness to launch with (control panel) and whose runs to browse:
  // Agent SDK / mini-swe / Codex, plus the eharness / vla / imagine lines
  // (runs live under outputs/beta-coding-agent / beta-react-harness /
  // beta-codex-agent). Live mode is SDK-runner-only.
  const [harness, setHarness] = usePersistentState<
    "claude-sdk" | "mini-swe" | "codex" | "eharness" | "vla" | "imagine"
  >("agentcanvas.coding.harness", "claude-sdk");
  // launch model, constrained to the selected harness's board options; a
  // persisted value from another harness (or a harness without a board list)
  // falls back to the harness's first option
  const modelOpts = MODEL_OPTIONS[harness] ?? MODEL_OPTIONS["claude-sdk"];
  const model = modelOpts.some((o) => o.value === modelRaw)
    ? modelRaw
    : modelOpts[0].value;
  const [runsList, setRunsList] = useState<RunInfo[]>([]);
  const [browseRun, setBrowseRun] = usePersistentState<string | null>(
    "agentcanvas.coding.browseRun",
    null,
  );
  const [browseEpisodes, setBrowseEpisodes] = useState<EpisodeSummary[]>([]);
  const [browseStarted, setBrowseStarted] = useState<number[]>([]);

  const offsetRef = useRef(0);
  // Reset key is run+episode: a new run reuses episode indices (a fresh run's
  // episode 0 must not inherit the old run's log offset).
  const shownKeyRef = useRef<string | null>(null);
  const runRef = useRef<string | null>(null);
  const logBoxRef = useRef<HTMLDivElement | null>(null);
  const logContentRef = useRef<HTMLDivElement | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  // stats.html report panel (charts + tables generated after each run's metrics)
  const [showStats, setShowStats] = useState(false);

  const running = status?.state === "running" || status?.state === "starting";
  const shownEpisode =
    mode === "browse"
      ? (viewEpisode ?? (browseStarted.length ? browseStarted[0] : null))
      : (viewEpisode ??
        status?.active_episode ??
        (status?.started_episodes?.length
          ? status.started_episodes[status.started_episodes.length - 1]
          : null));

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const st: RunStatus = await (await fetch("/api/coding-agent/status")).json();
        if (cancelled) return;
        setStatus(st);

        if (mode === "live" && runRef.current !== st.run_name) {
          runRef.current = st.run_name;
          setViewEpisode(null);
          setZoomFrame(null);
        }

        let run: string | null;
        let ep: number | null;
        if (mode === "browse") {
          run = browseRun;
          ep = viewEpisode ?? (browseStarted.length ? browseStarted[0] : null);
        } else {
          run = st.run_name;
          ep =
            viewEpisode ??
            st.active_episode ??
            (st.started_episodes.length
              ? st.started_episodes[st.started_episodes.length - 1]
              : null);
        }
        if (run == null || ep == null) {
          // no shown run (e.g. mid harness-switch, browseRun reset to null):
          // clear the previous run's log instead of letting it linger
          if (shownKeyRef.current !== null) {
            shownKeyRef.current = null;
            offsetRef.current = 0;
            setLines([]);
            setFrames([]);
            setZoomFrame(null);
          }
          return;
        }

        const shownKey = `${run}:${ep}`;
        if (shownKeyRef.current !== shownKey) {
          shownKeyRef.current = shownKey;
          offsetRef.current = 0;
          setLines([]);
          setFrames([]);
          setZoomFrame(null);
          setHarnessData(null);
          setSnap(null);          // never show the previous selection's images
          setTopdownDead(false);
          setAmapDead(false);
        }

        const src = mode === "browse" ? harness : (st.config?.harness ?? "claude-sdk");
        // §14.14 late-response guard: a slow tick's fetches may land AFTER the
        // selection moved on (a live-mode run change does not recreate this
        // effect, so `cancelled` alone cannot catch it). Every setState below
        // re-checks that the selection this tick fetched for is still shown.
        const stale = () => cancelled || shownKeyRef.current !== shownKey;
        const [logRes, framesRes] = await Promise.all([
          fetch(
            `/api/coding-agent/runs/${run}/episode/${ep}/textlog?offset=${offsetRef.current}&source=${src}`,
          ),
          fetch(`/api/coding-agent/runs/${run}/episode/${ep}/frames?source=${src}`),
        ]);
        const logData = await logRes.json();
        const framesData = await framesRes.json();
        if (stale()) return;

        if (logData.lines.length > 0) {
          offsetRef.current = logData.next_offset;
          setLines((prev) => [...prev, ...logData.lines]);
        }
        setFrames(framesData.frames);

        // eharness organ readout rides the same 1 Hz tick (browse-only source)
        if (src === "eharness") {
          const hd = await (
            await fetch(
              `/api/coding-agent/runs/${run}/episode/${ep}/harness?source=${src}`,
            )
          ).json();
          if (!stale()) setHarnessData(hd);
          // §6: read the ATOMIC snapshot first, then fetch exactly the files
          // it names — never a mix of independently-updating latest files
          try {
            const sn = await fetch(
              `/api/coding-agent/runs/${run}/episode/${ep}/snapshot?source=${src}`,
            );
            const snData = sn.ok ? await sn.json() : null;
            // an inconsistent snapshot (mid-write, files missing) must not
            // replace the last good one — fetching its missing files would
            // 404 and retire the map panels; keep showing the previous
            // consistent set until the writer catches up
            if (!stale() && snData?.consistent !== false) setSnap(snData);
          } catch {
            if (!stale()) setSnap(null);
          }
        }
      } catch {
        /* backend unreachable — keep polling */
      }
    };

    tick();
    const interval = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [viewEpisode, mode, browseRun, browseStarted, harness]);

  const loadRuns = async (source?: "claude-sdk" | "mini-swe" | "codex" | "eharness" | "vla" | "imagine") => {
    try {
      const src = source ?? harness;
      const data = await (await fetch(`/api/coding-agent/runs?source=${src}`)).json();
      setRunsList(data.runs ?? []);
      if (data.runs?.length) setBrowseRun((r) => r ?? data.runs[0].name);
    } catch {
      /* backend unreachable */
    }
  };

  // On mount, if a persisted refresh landed us back in browse mode, repopulate
  // the run list (loadRuns keeps the restored browseRun via its `r ?? …` guard).
  // Live mode needs nothing here — the status poll below drives it.
  useEffect(() => {
    if (mode === "browse") loadRuns(harness);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // per-run episode outcomes for the browse selector badges
  useEffect(() => {
    if (mode !== "browse" || !browseRun) {
      // no browsed run (mid harness-switch / back to live): drop the previous
      // run's episode badges instead of letting them linger
      setBrowseEpisodes([]);
      setBrowseStarted([]);
      return;
    }
    // switching runs: blank the badges until the new run's summary arrives
    setBrowseEpisodes([]);
    setBrowseStarted([]);
    let cancelled = false;
    (async () => {
      try {
        const d = await (
          await fetch(`/api/coding-agent/runs/${browseRun}/summary?source=${harness}`)
        ).json();
        if (cancelled) return;
        setBrowseEpisodes(d.episodes ?? []);
        setBrowseStarted(d.started_episodes ?? []);
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [mode, browseRun, harness]);

  // stick to bottom unless the user scrolled up
  useEffect(() => {
    const box = logBoxRef.current;
    if (!box) return;
    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
    if (nearBottom) box.scrollTop = box.scrollHeight;
  }, [lines]);

  const start = async () => {
    setStartError(null);
    // parse "k=v k=v" (whitespace-separated) into the extra knob dict
    const extra: Record<string, string> = {};
    for (const pair of extraText.trim().split(/\s+/).filter(Boolean)) {
      const eq = pair.indexOf("=");
      if (eq <= 0) {
        setStartError(`bad extra knob ${JSON.stringify(pair)} — expected k=v`);
        return;
      }
      extra[pair.slice(0, eq)] = pair.slice(eq + 1);
    }
    const res = await fetch("/api/coding-agent/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episodes,
        split,
        max_turns: maxTurns,
        model: model.trim() || null,
        harness,
        condition,
        tier,
        extra,
      }),
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => null))?.detail;
      setStartError(detail ?? `start failed (${res.status})`);
      return;
    }
    setViewEpisode(null);
    setZoomFrame(null);
  };

  const stop = async () => {
    await fetch("/api/coding-agent/stop", { method: "POST" });
  };

  const run = mode === "browse" ? browseRun : status?.run_name;
  // the shown run's log source: browse follows the selector; live follows the
  // harness the RUNNING run was launched with (not the selector, which the
  // user may have flipped since pressing Run)
  const shownSrc = mode === "browse" ? harness : (status?.config?.harness ?? "claude-sdk");
  const frameUrl = (name: string) =>
    `/api/coding-agent/runs/${run}/episode/${shownEpisode}/frame/${name}?source=${shownSrc}`;
  // cache-buster ticks with the run's own progress so the map refreshes as the
  // organ redraws it; falls back through whichever counter this backend serves
  // §6: with a snapshot the images are the VERSIONED files it names (cache
  // key = obs/map version); the latest.png fallbacks only serve runs from
  // before the atomic-snapshot change.
  const snapKey = snap ? `${snap.obs_id}.${snap.map_version}` : null;
  const topdownUrl = snap
    ? `${frameUrl(snap.topdown_file)}&v=${snapKey}`
    : `${frameUrl("topdown_latest.png")}&t=` +
      `${harnessData?.depth?.obs ?? harnessData?.heartbeat?.steps_used ?? 0}`;
  const amapUrl = snap
    ? `${frameUrl(snap.accumulated_map_file)}&v=${snapKey}`
    : `${frameUrl("map_latest.png")}&t=` +
      `${harnessData?.depth?.obs ?? harnessData?.heartbeat?.steps_used ?? 0}`;
  const amapModelUrl = snap
    ? `${frameUrl(snap.model_map_file)}&v=${snapKey}`
    : `${frameUrl("map_model_latest.png")}&t=` +
      `${harnessData?.depth?.obs ?? harnessData?.heartbeat?.steps_used ?? 0}`;
  // a retired panel must come back when you switch to a run that HAS a map,
  // otherwise one 404 hides it for the rest of the session
  if (shownRunEp !== `${run}#${shownEpisode}`) {
    setShownRunEp(`${run}#${shownEpisode}`);
    if (topdownDead) setTopdownDead(false);
    if (amapDead) setAmapDead(false);
  }
  const epList = mode === "browse" ? browseEpisodes : (status?.episodes ?? []);
  const epSummary = (i: number) => epList.find((e) => e.index === i);
  const selEpisodes = mode === "browse" ? browseStarted : (status?.started_episodes ?? []);
  const browseInfo = runsList.find((r) => r.name === browseRun);

  // Export the currently shown episode's log as ONE tall PDF page:
  // rasterize the full log content (logContentRef has natural height, so we
  // capture everything — not just the scroll-clipped viewport) to a canvas,
  // then wrap it in a single A4-width jsPDF page (height scales with content).
  // Capture scale is capped so a long episode stays under the browser canvas
  // (~16k px per side) limit before it is mapped onto the A4-wide page.
  const exportPdf = async () => {
    const el = logContentRef.current;
    if (!el || exporting) return;
    setExportError(null);
    setExporting(true);
    try {
      // Fit the capture to a narrow, A4-proportioned column so text isn't
      // shrunk when the (often wide) log panel is squeezed to A4 width: reflow
      // the content to EXPORT_W and capture at that width. The on-screen panel
      // is untouched — the width is applied only to html2canvas's offscreen
      // clone (via onclone), never to the live DOM.
      const EXPORT_W = 620;
      // Measure the reflowed height synchronously (set → read → restore within
      // one tick, so the browser never paints the narrow state → no flash) to
      // pick a scale that keeps the canvas under the ~16k px browser limit.
      const prevW = el.style.width;
      const prevMax = el.style.maxWidth;
      el.style.width = `${EXPORT_W}px`;
      el.style.maxWidth = `${EXPORT_W}px`;
      const reflowH = el.scrollHeight;
      el.style.width = prevW;
      el.style.maxWidth = prevMax;
      const scale = Math.min(2, 16000 / reflowH);
      const canvas = await html2canvas(el, {
        backgroundColor: "#111827", // gray-900 — matches the log panel
        scale,
        useCORS: true,
        imageTimeout: 15000,
        width: EXPORT_W,
        windowWidth: EXPORT_W,
        onclone: (doc) => {
          const node = doc.querySelector<HTMLElement>("[data-log-export]");
          if (node) {
            node.style.width = `${EXPORT_W}px`;
            node.style.maxWidth = `${EXPORT_W}px`;
          }
        },
      });
      const imgData = canvas.toDataURL("image/jpeg", 0.92);
      // One long page at A4 width (210 mm); height scales with the content so
      // it stays a single long strip rather than a fixed A4 sheet.
      const A4_W_MM = 210;
      const pageH = (A4_W_MM * canvas.height) / canvas.width;
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: [A4_W_MM, pageH],
      });
      pdf.addImage(imgData, "JPEG", 0, 0, A4_W_MM, pageH);
      const fname = `${run ?? "log"}_ep${shownEpisode ?? 0}.pdf`.replace(
        /[^\w.-]+/g,
        "_",
      );
      pdf.save(fname);
    } catch (e) {
      setExportError(e instanceof Error ? e.message : "export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto bg-gray-950 p-3 text-gray-200">
      {/* ── control panel ── */}
      <div className="flex flex-wrap items-center gap-3 rounded border border-gray-800 bg-gray-900 px-3 py-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-blue-400">
          <Bot size={18} />
          Coding-Agent Monitor
        </div>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          harness
          <select
            value={harness}
            onChange={(e) => setHarness(e.target.value as typeof harness)}
            disabled={running}
            className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          >
            <option value="claude-sdk">Claude SDK</option>
            <option value="mini-swe">mini-swe-agent</option>
            <option value="codex">Codex CLI</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          model
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={running}
            className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          >
            {modelOpts.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          condition
          <select
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
            disabled={running}
            title="the MIP paper conditions"
            className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          >
            <option value="bare">bare</option>
            <option value="wp">wp</option>
            <option value="hybrid">hybrid</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          tier
          <select
            value={tier}
            onChange={(e) => setTier(e.target.value)}
            disabled={running}
            title="effort tier — expands to the per-(harness, model) knobs the std board uses"
            className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          >
            <option value="default">default</option>
            <option value="max">max</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          split
          <select
            value={split}
            onChange={(e) => setSplit(e.target.value)}
            disabled={running}
            className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          >
            <option value="rand100">rand100</option>
            <option value="val_unseen">val_unseen</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          episodes
          <input
            value={episodes}
            onChange={(e) => setEpisodes(e.target.value)}
            disabled={running}
            className="w-20 rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          />
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          max-turns
          <input
            type="number"
            value={maxTurns}
            onChange={(e) => setMaxTurns(Number(e.target.value))}
            disabled={running}
            className="w-16 rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          />
        </label>
        <label className="flex items-center gap-1 text-xs text-gray-400">
          extra
          <input
            value={extraText}
            onChange={(e) => setExtraText(e.target.value)}
            disabled={running}
            placeholder="k=v k=v"
            title="harness extra knobs, e.g. effort=xhigh thinking=adaptive api_base=…"
            className="w-32 rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
          />
        </label>
        {running ? (
          <button
            onClick={stop}
            className="flex items-center gap-1 rounded bg-red-700 px-3 py-1 text-xs font-medium text-white hover:bg-red-600"
          >
            <Square size={12} /> Stop
          </button>
        ) : (
          <button
            onClick={start}
            className="flex items-center gap-1 rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-500"
          >
            <Play size={12} /> Run
          </button>
        )}
        <span
          className={clsx(
            "rounded px-1.5 py-0.5 text-xs font-medium",
            status?.state === "running" && "bg-green-600/30 text-green-300",
            status?.state === "starting" && "bg-yellow-600/30 text-yellow-300",
            status?.state === "error" && "bg-red-600/30 text-red-300",
            (status?.state === "finished" || status?.state === "stopped") &&
              "bg-blue-600/30 text-blue-300",
            (!status || status.state === "idle") && "bg-gray-700/50 text-gray-400",
          )}
        >
          {status?.state ?? "…"}
        </span>
        {status?.aggregate && (
          <span className="text-xs text-gray-400">
            SR {status.aggregate.success?.toFixed(2)} · SPL{" "}
            {status.aggregate.spl?.toFixed(2)} · stop{" "}
            {status.aggregate.stop_rate?.toFixed(2)} ·{" "}
            {status.aggregate.episode_count} eps
          </span>
        )}
        {(startError || status?.error) && (
          <span className="text-xs text-red-400">{startError ?? status?.error}</span>
        )}
      </div>

      {/* ── source selector: live run vs stored logs ── */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <div className="flex overflow-hidden rounded border border-gray-700">
          <button
            onClick={() => {
              setMode("live");
              setViewEpisode(null);
            }}
            className={clsx(
              "px-2 py-0.5",
              mode === "live"
                ? "bg-blue-600/30 text-blue-300"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Live
          </button>
          <button
            onClick={() => {
              setMode("browse");
              setViewEpisode(null);
              loadRuns();
            }}
            className={clsx(
              "px-2 py-0.5",
              mode === "browse"
                ? "bg-blue-600/30 text-blue-300"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Logs
          </button>
        </div>
        {mode === "browse" && (
          <div className="flex overflow-hidden rounded border border-gray-700">
            {(
              [
                ["claude-sdk", "Claude SDK"],
                ["mini-swe", "mini-swe-agent"],
                ["codex", "Codex CLI"],
                ["eharness", "Embodied Harness"],
                ["vla", "VLA Harness"],
                ["imagine", "ImagineVLN"],
              ] as const
            ).map(([h, label]) => (
              <button
                key={h}
                onClick={() => {
                  if (harness === h) return;
                  setHarness(h);
                  setRunsList([]);
                  setBrowseRun(null);
                  setViewEpisode(null);
                  loadRuns(h);
                }}
                className={clsx(
                  "px-2 py-0.5",
                  harness === h
                    ? "bg-purple-600/30 text-purple-300"
                    : "bg-gray-800 text-gray-400 hover:text-gray-200",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        {mode === "browse" && (
          <>
            <span className="text-gray-500">run:</span>
            <select
              value={browseRun ?? ""}
              onChange={(e) => {
                setBrowseRun(e.target.value);
                setViewEpisode(null);
              }}
              className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
            >
              {runsList.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
            {browseInfo && (
              <span className="text-gray-500">
                {browseInfo.model ?? "default-model"} ·{" "}
                {browseInfo.skill ? `skill:${browseInfo.skill}` : "no-skill"} ·{" "}
                {browseInfo.success != null
                  ? `SR ${browseInfo.success.toFixed(2)}`
                  : "no summary"}{" "}
                · {browseInfo.episodes.length} eps
              </span>
            )}
            <button
              onClick={() => loadRuns()}
              title="refresh run list"
              className="rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-gray-400 hover:text-gray-200"
            >
              ↻
            </button>
          </>
        )}
      </div>

      {/* ── episode selector ── */}
      {selEpisodes.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 text-xs">
          <span className="mr-1 text-gray-500">episode:</span>
          {selEpisodes.map((i) => {
            const s = epSummary(i);
            // Two independent questions, answered in order:
            //  1. Does this episode count at all? A rate-limit / "limit
            //     exceeded" casualty errored WITHOUT taking a single navigation
            //     step (error + env_steps 0), so it never attempted the task —
            //     bare ⚠, and excluded from SR (see driver.is_scored).
            //  2. If it counts, what happened — and did it end abnormally? A
            //     timeout that was still evaluated has a real nav result, so
            //     show the result and append ⚠ rather than hiding it behind the
            //     error tag. A turn-exhausted run (env_steps > 0) is a genuine
            //     ❌ and stays in the denominator.
            const limitExceeded =
              s != null && s.error != null && (s.env_steps ?? 0) === 0;
            const badge =
              s == null
                ? "…"
                : limitExceeded
                  ? "⚠"
                  : s.success != null
                    ? s.error
                      ? s.success
                        ? "✅⚠"
                        : "❌⚠"
                      : s.success
                        ? "✅"
                        : "❌"
                    : s.error
                      ? "⚠"
                      : "…";
            return (
              <button
                key={i}
                // one tooltip for the two ways an episode can carry an error,
                // matching the badge above: excluded from SR vs scored but
                // ended abnormally
                title={
                  s?.error
                    ? limitExceeded
                      ? `not attempted (excluded from SR): ${s.error}`
                      : `ended abnormally: ${s.error}`
                    : undefined
                }
                onClick={() => setViewEpisode(i)}
                className={clsx(
                  "rounded border px-2 py-0.5",
                  shownEpisode === i
                    ? "border-blue-500 bg-blue-600/20 text-blue-300"
                    : "border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200",
                )}
              >
                {i} {badge}
              </button>
            );
          })}
          {mode === "live" && viewEpisode != null && (
            <button
              onClick={() => setViewEpisode(null)}
              className="ml-1 rounded border border-gray-700 bg-gray-800 px-2 py-0.5 text-gray-400 hover:text-gray-200"
            >
              follow active
            </button>
          )}
        </div>
      )}

      {/* ── stats report (charts + tables generated after each run's metrics) ── */}
      {showStats && run && (
        <div className="flex h-[70vh] shrink-0 flex-col overflow-hidden rounded border border-gray-800">
          <div className="flex items-center justify-between border-b border-gray-800 bg-gray-900 px-3 py-1.5 text-xs font-semibold text-gray-400">
            <span>Stats — {run}</span>
            <a
              href={`/api/coding-agent/runs/${run}/stats?source=${shownSrc}`}
              target="_blank"
              rel="noreferrer"
              className="font-normal text-blue-400 hover:text-blue-300"
            >
              open in new tab ↗
            </a>
          </div>
          <iframe
            src={`/api/coding-agent/runs/${run}/stats?source=${shownSrc}`}
            title={`stats — ${run}`}
            className="min-h-0 flex-1 bg-white"
          />
        </div>
      )}

      {/* ── eharness organ panel: the state block the model actually sees,
             the harness-written heartbeat, event keyframes, receipts ── */}
      {harnessData && (harnessData.state || harnessData.heartbeat) && (
        <div className="shrink-0 space-y-2 rounded border border-gray-800 bg-gray-900 p-2 text-xs">
          <div className="flex items-center justify-between font-semibold text-gray-400">
            <button
              onClick={() => setShowHarness((v) => !v)}
              className="flex items-center gap-1 hover:text-gray-200"
              title={showHarness ? "collapse" : "expand"}
            >
              <span>{showHarness ? "▾" : "▸"}</span>
              <span>🧠 Harness — 状态块 · 心跳 · 关键帧</span>
            </button>
            {harnessData.heartbeat && (
              <span className="font-normal text-gray-500">
                💓 {harnessData.heartbeat.status ?? "—"}
                {harnessData.heartbeat.subgoal
                  ? ` · ${harnessData.heartbeat.subgoal.slice(0, 50)}`
                  : ""}
                {" · steps "}
                {harnessData.heartbeat.steps_used ?? 0}
                {harnessData.heartbeat.steps_budget
                  ? `/${harnessData.heartbeat.steps_budget}`
                  : ""}
                {" · guards "}
                {harnessData.heartbeat.guard_trips ?? 0}
              </span>
            )}
          </div>
          {showHarness && (
          <div className="max-h-52 space-y-2 overflow-y-auto pr-1">
          {harnessData.state && (
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
              <div className="space-y-0.5">
                {(harnessData.state.sub_instructions ?? []).map((si, i) => {
                  const cur = harnessData.state?.cursor ?? 0;
                  return (
                    <div
                      key={i}
                      className={clsx(
                        "truncate rounded px-1.5 py-0.5",
                        i < cur
                          ? "text-gray-600 line-through"
                          : i === cur
                            ? "bg-blue-600/20 font-semibold text-blue-300"
                            : "text-gray-500",
                      )}
                      title={si}
                    >
                      {i === cur ? "▸ " : "  "}
                      [{i + 1}] {si}
                    </div>
                  );
                })}
              </div>
              <div className="space-y-1">
                {harnessData.state.current_place && (
                  <div className="text-gray-300">
                    📍 {harnessData.state.current_place}
                  </div>
                )}
                {Object.keys(harnessData.state.landmarks ?? {}).length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {Object.keys(harnessData.state.landmarks ?? {}).map((n) => (
                      <span
                        key={n}
                        className="rounded bg-emerald-600/20 px-1.5 py-0.5 text-emerald-300"
                      >
                        🚩 {n}
                      </span>
                    ))}
                  </div>
                )}
                {(harnessData.state.negative_facts ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {(harnessData.state.negative_facts ?? []).map((f, i) => (
                      <span
                        key={i}
                        className="rounded bg-red-600/15 px-1.5 py-0.5 text-red-300"
                        title={f.verified ? "verified" : "unverified"}
                      >
                        🚫 {f.fact}
                        {!f.verified && " ?"}
                      </span>
                    ))}
                  </div>
                )}
                {(harnessData.state.last_action || harnessData.state.expectation) && (
                  <div className="text-gray-400">
                    {harnessData.state.last_action && (
                      <span>🦶 just did: {harnessData.state.last_action}</span>
                    )}
                    {harnessData.state.expectation && (
                      <span className="ml-2 text-amber-300/80">
                        🔮 expected: {harnessData.state.expectation.slice(0, 70)}
                      </span>
                    )}
                  </div>
                )}
                {harnessData.state.delegation && (
                  <div className="text-purple-300">
                    📦 delegating: {harnessData.state.delegation.subgoal.slice(0, 60)} (
                    {harnessData.state.delegation.used}/
                    {harnessData.state.delegation.budget} steps)
                  </div>
                )}
              </div>
            </div>
          )}
          {/* Depth organ readout. Deliberately keyed off the IMAGE, not off the
              harness JSON: the top-down map is served by the frame endpoint that
              already exists, and its numbers are burned into the picture, so the
              panel works without a backend restart. The JSON (when a newer
              backend supplies it) only enriches the text beside it. */}
          {/* hide-not-unmount: a dead flag from one 404 must not unmount the
              img — a mounted img reloads when its src advances on the next
              good snapshot and onLoad revives the panel by itself */}
          {(
            <div
              style={topdownDead ? { display: "none" } : undefined}
              className="rounded border border-cyan-900/60 bg-cyan-950/20 p-2"
            >
              <div className="mb-1 flex items-center justify-between text-cyan-300">
                <span>🗺️ 深度器官 · 实时俯视图（机器人在底部中央，朝上）</span>
                <span className="text-[10px] text-gray-500">
                  模型看不到这些 · 仅供人观察
                </span>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setZoomFrame(topdownUrl)}
                  className="shrink-0"
                  title="点击放大"
                >
                  <img
                    src={topdownUrl}
                    alt="top-down map"
                    onError={() => setTopdownDead(true)}
                    onLoad={() => setTopdownDead(false)}
                    className="h-52 rounded border border-gray-800 bg-black object-contain"
                  />
                </button>
                <div className="min-w-0 flex-1 space-y-0.5 text-[11px]">
                  {harnessData.depth ? (
                    <>
                      <div className="text-gray-400">
                        正前方可走{" "}
                        <span className="text-cyan-300">{harnessData.depth.ahead_m}m</span>
                        {" · 最开阔 "}
                        <span className="text-cyan-300">
                          {(harnessData.depth.widest_bearing_deg ?? 0) > 0 ? "左" : "右"}
                          {Math.abs(harnessData.depth.widest_bearing_deg ?? 0)}° @{" "}
                          {harnessData.depth.widest_m}m
                        </span>
                      </div>
                      <div className="text-gray-500">
                        地图 free {harnessData.depth.free_pct}% · occ{" "}
                        {harnessData.depth.occupied_pct}% · unknown{" "}
                        {harnessData.depth.unknown_pct}% · 格 {harnessData.depth.cell_m}m ·
                        上限 {harnessData.depth.range_cap_m}m
                      </div>
                      <div className="text-gray-600">
                        深度单位 {harnessData.depth.depth_units} · 相机离地{" "}
                        {harnessData.depth.floor_below_camera_m}m · goto ×
                        {harnessData.depth.gotos ?? 0} · SAM3 {harnessData.depth.detector}
                      </div>
                      {(harnessData.depth.candidates ?? []).map((c) => (
                        <div key={c.n} className="text-gray-300">
                          <span className="text-cyan-400">{c.n}</span>{" "}
                          <span className="text-gray-500">[{c.kind}]</span> {c.where}
                          <span className="text-gray-600">
                            {" "}
                            → {c.env_steps} 步
                            {c.squeeze_m != null ? ` · 最窄处 ${c.squeeze_m}m` : ""}
                          </span>
                        </div>
                      ))}
                      {(harnessData.depth.landmarks ?? []).length > 0 && (
                        <div className="text-emerald-300">
                          👁 SAM3:{" "}
                          {(harnessData.depth.landmarks ?? [])
                            .map(
                              (l) =>
                                `${l.phrase} ${l.bearing_deg > 0 ? "左" : "右"}${Math.abs(l.bearing_deg)}° ${l.distance_m}m`,
                            )
                            .join(" · ")}
                        </div>
                      )}
                      {harnessData.depth.landmark_ledger && (
                        <details className="text-gray-600">
                          <summary className="cursor-pointer select-none hover:text-gray-400">
                            地标账本（喂裁判，不给模型）
                          </summary>
                          {Object.entries(harnessData.depth.landmark_ledger).map(
                            ([k, v]) => (
                              <div key={k} className="pl-2">
                                <span className="text-gray-500">{k}:</span> {v}
                              </div>
                            ),
                          )}
                        </details>
                      )}
                    </>
                  ) : (
                    <div className="text-gray-500">
                      读数已烧录在图片下方的字幕条里。重启后端（picks up the{" "}
                      <code>depth</code> key）后这里会显示可点开的结构化版本。
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          {(
            <div
              style={amapDead ? { display: "none" } : undefined}
              className="rounded border border-indigo-900/60 bg-indigo-950/20 p-2"
            >
              <div className="mb-1 flex items-center justify-between text-indigo-300">
                <span title="2.5D：深度压成占据+语义俯视层；楼梯/多层/悬空障碍不在此图的表达范围">
                  🧭 2.5D accumulated top-down map（左：锚定全景·给人看 / 右：模型每轮收到的 IMAGE 2）
                </span>
                <span className="text-[10px] text-gray-500">
                  {snap
                    ? `snapshot obs#${snap.obs_id} · step ${snap.env_step} · map v${snap.map_version}` +
                      (snap.identity
                        ? ` · ${[
                            snap.identity.executor,
                            snap.identity.run_id && snap.identity.episode
                              ? `${snap.identity.run_id}/${snap.identity.episode}`
                              : snap.identity.run_id || snap.identity.episode,
                            snap.identity.action_id,
                          ]
                            .filter(Boolean)
                            .join(" · ")}`
                        : "") +
                      (snap.consistent ? "" : " · ⚠ 文件不齐")
                    : "latest 文件（旧 run 兼容）"}{" "}
                  · 右图机器人居中朝上 · 实心=当前看见 · 虚环=记忆 · 琥珀=瞥见未验证
                </span>
              </div>
              <div className="flex gap-3">
                <button onClick={() => setZoomFrame(amapUrl)} className="shrink-0" title="点击放大">
                  <img
                    src={amapUrl}
                    alt="accumulated map"
                    onError={() => setAmapDead(true)}
                    onLoad={() => setAmapDead(false)}
                    className="h-64 rounded border border-gray-800 bg-black object-contain"
                  />
                </button>
                <button onClick={() => setZoomFrame(amapModelUrl)} className="shrink-0" title="点击放大">
                  <img
                    src={amapModelUrl}
                    alt="model-facing map"
                    onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                    onLoad={(e) => ((e.target as HTMLImageElement).style.display = "")}
                    className="h-64 rounded border border-gray-800 bg-black object-contain"
                  />
                </button>
              </div>
            </div>
          )}
          {harnessData.state?._rendered && (
            <details className="text-gray-500">
              <summary className="cursor-pointer select-none hover:text-gray-300">
                模型看到的 [STATE] 原文
              </summary>
              <pre className="mt-1 whitespace-pre-wrap rounded bg-gray-950 p-2 text-[11px] text-gray-400">
                {harnessData.state._rendered}
              </pre>
            </details>
          )}
          {harnessData.keyframes.length > 0 && (
            <div className="flex gap-2 overflow-x-auto pb-1">
              {harnessData.keyframes.map((kf) => (
                <button
                  key={kf.idx}
                  onClick={() => kf.png && setZoomFrame(kf.png)}
                  className="shrink-0 text-left"
                  title={kf.events.join(" · ")}
                >
                  {kf.png ? (
                    <img
                      src={frameUrl(kf.png)}
                      alt={`kf ${kf.idx}`}
                      className="h-20 rounded border border-gray-700"
                    />
                  ) : (
                    <div className="flex h-20 w-28 items-center justify-center rounded border border-gray-700 text-gray-600">
                      #{kf.idx}
                    </div>
                  )}
                  <div className="w-28 truncate text-[10px] text-gray-500">
                    #{kf.idx} {kf.events.slice(-1)[0] ?? ""}
                  </div>
                </button>
              ))}
            </div>
          )}
          {(harnessData.segments ?? []).length > 0 && (
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-wide text-gray-600">
                走过的段落（集内长期记忆 · recall("segment N") 可回放）
              </div>
              {(harnessData.segments ?? []).map((sg) => (
                <button
                  key={sg.seg}
                  onClick={() => setZoomFrame(sg.png)}
                  className="block w-full text-left"
                  title={`route: ${sg.route || "—"} · moved: ${sg.motion || "—"}`}
                >
                  <div className="truncate text-[11px] text-gray-400">
                    🎞 seg {sg.seg} · {sg.label}
                    {sg.route ? ` — ${sg.route.slice(0, 60)}` : ""}
                  </div>
                  <img
                    src={frameUrl(sg.png)}
                    alt={`segment ${sg.seg}`}
                    className="max-h-24 rounded border border-gray-700"
                  />
                </button>
              ))}
            </div>
          )}
          {harnessData.receipts.length > 0 && (
            <div className="space-y-0.5">
              {harnessData.receipts.map((r, i) => (
                <div key={i} className="truncate text-gray-400">
                  📦 {r.claim === "reached" ? "✅" : "⚠"} {r.claim} ·{" "}
                  {r.subgoal.slice(0, 70)} · {r.steps_used ?? "?"} steps
                  {r.verdict ? ` · ⚖ ${r.verdict}` : ""}
                  {r.not_done ? ` · not done: ${r.not_done.slice(0, 60)}` : ""}
                </div>
              ))}
            </div>
          )}
          </div>
          )}
        </div>
      )}

      {/* ── unified log (frames embedded inline at their observe calls) ── */}
      <div className="flex min-h-[65vh] flex-1 flex-col rounded border border-gray-800 bg-gray-900">
        <div className="flex items-center justify-between border-b border-gray-800 px-3 py-1.5 text-xs font-semibold text-gray-400">
          <span>
            {mode === "browse" ? `Log — ${run ?? "…"}` : "Live Log"}
            {shownEpisode != null && ` — episode ${shownEpisode}`}
          </span>
          <div className="flex items-center gap-2">
            {exportError && (
              <span className="font-normal text-red-400">{exportError}</span>
            )}
            <button
              onClick={() => setShowStats((v) => !v)}
              disabled={!run}
              title="full statistics report for this run (charts + tables, generated after metrics)"
              className={clsx(
                "flex items-center gap-1 rounded border px-2 py-0.5 font-normal disabled:cursor-not-allowed disabled:opacity-40",
                showStats
                  ? "border-purple-500 bg-purple-600/20 text-purple-300"
                  : "border-gray-700 bg-gray-800 text-gray-300 hover:text-gray-100",
              )}
            >
              <BarChart3 size={12} />
              stats
            </button>
            <button
              onClick={exportPdf}
              disabled={exporting || lines.length === 0}
              title="export this episode's log as one long PDF page"
              className="flex items-center gap-1 rounded border border-gray-700 bg-gray-800 px-2 py-0.5 font-normal text-gray-300 hover:text-gray-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Download size={12} />
              {exporting ? "exporting…" : "PDF"}
            </button>
          </div>
        </div>
        <div ref={logBoxRef} className="min-h-0 flex-1 overflow-y-auto p-2 font-mono text-xs">
          <div ref={logContentRef} data-log-export>
          {lines.length === 0 && (
            <div className="p-4 text-gray-600">no events yet</div>
          )}
          {(() => {
            // Frame pairing is data-driven and keyed on the frames ACTUALLY on
            // disk, not on the transcript's image-block count. Frames are named
            // obs_<NNNN>_step<SSS>[ _depth ].png; grouping by the obs index gives
            // one group per viewpoint — an observe writes one group (RGB, plus a
            // paired _depth frame on post-fix runs), a look_around writes eight
            // (one RGB each). The cursor walks GROUPS, so it advances by the
            // viewpoints a tool produced, which equals the groups that tool wrote
            // in BOTH old runs (RGB only) and new runs (RGB+depth). That stops an
            // old observe — whose result carried two image blocks but only ever
            // wrote one frame — from pulling the NEXT observe's RGB into its depth
            // slot and starving every later step of an image.
            const IMG = "<image elided>";
            const resultByToolUse: Record<string, string[]> = {};
            for (const l of lines) {
              if (l.kind === "tool_result" && typeof l.tool_use_id === "string") {
                resultByToolUse[l.tool_use_id] =
                  (l.texts as string[] | undefined) ?? [];
              }
            }
            // Pair each tool_use to its result. Unified-driver logs carry
            // id/tool_use_id; legacy runs (e.g. fable50_bare) emit neither, so
            // fall back to positional pairing — the next tool_result after this
            // tool_use, consumed once (legacy logs strictly alternate use/result).
            const resultByLineIdx: Record<number, string[]> = {};
            const orderedResults: { idx: number; texts: string[] }[] = [];
            lines.forEach((l, idx) => {
              if (l.kind === "tool_result")
                orderedResults.push({ idx, texts: (l.texts as string[] | undefined) ?? [] });
            });
            let resPtr = 0;
            lines.forEach((l, idx) => {
              if (l.kind !== "tool_use") return;
              const byId =
                typeof l.id === "string" ? resultByToolUse[l.id] : undefined;
              if (byId) {
                resultByLineIdx[idx] = byId;
                return;
              }
              while (resPtr < orderedResults.length && orderedResults[resPtr].idx <= idx)
                resPtr++;
              if (resPtr < orderedResults.length)
                resultByLineIdx[idx] = orderedResults[resPtr++].texts;
            });
            const groups: { rgb: string | null; depth: string | null }[] = [];
            const groupAt: Record<string, number> = {};
            for (const f of frames) {
              const m = f.match(/^obs_(\d+)_/);
              const key = m ? m[1] : f;
              if (!(key in groupAt)) {
                groupAt[key] = groups.length;
                groups.push({ rgb: null, depth: null });
              }
              if (f.includes("_depth")) groups[groupAt[key]].depth = f;
              else groups[groupAt[key]].rgb = f;
            }
            // tool_use line index → the frame names its own result declared.
            const framesByLineIdx: Record<number, string[]> = {};
            {
              const pending: number[] = [];
              lines.forEach((l, idx) => {
                if (l.kind === "tool_use") pending.push(idx);
                else if (l.kind === "tool_result" && Array.isArray(l.frames)) {
                  const owner = pending.length ? pending.pop()! : idx;
                  framesByLineIdx[owner] = l.frames as string[];
                }
              });
            }
            let groupCursor = 0;
            return lines.map((line, i) => {
              // A tool_use's viewpoints = image blocks in its paired result NOT
              // immediately preceded by another image block. observe's depth
              // block sits right after its RGB block → same viewpoint; each
              // look_around view is preceded by its text label → a new viewpoint.
              // Each viewpoint consumes one on-disk obs group, rendered as its
              // RGB tile plus a depth tile when that frame exists. A pending
              // tool_use (no result yet) is always the latest line, so earlier
              // lines never desync and tiles fill in on the next poll.
              const tiles: { url: string | null; label: string | null }[] = [];
              let nViews = 0;
              // Data-driven path: a driver that knows which frames its call
              // wrote says so on the tool_result (`frames: [...]`). Nothing to
              // infer, no cursor to keep in sync — the VLA harness writes one
              // group per dispatch (leg start/middle/end plus the side views),
              // so a segment's whole image history renders at its own call.
              const explicit = framesByLineIdx[i];
              if (explicit?.length) {
                for (const name of explicit) {
                  const m = name.match(/^obs_\d+_(.+)\.[a-z]+$/i);
                  const tag = m ? m[1] : "";
                  tiles.push({
                    url: name,
                    label:
                      tag === "left" ? "left" :
                      tag === "right" ? "right" :
                      tag === "0" ? "leg start" :
                      tag === "2" ? "leg end" :
                      tag ? `during ${tag}` : null,
                  });
                }
              } else if (line.kind === "tool_use") {
                const res = resultByLineIdx[i];
                if (res) {
                  const labels: string[] = [];
                  for (let j = 0; j < res.length; j++) {
                    if (res[j] !== IMG || res[j - 1] === IMG) continue;
                    const prev = j > 0 ? res[j - 1] : undefined;
                    // §10.6: every result now carries the accumulated map as
                    // a SECOND image. It is not a camera viewpoint and has no
                    // obs_* group on disk — counting it advanced the group
                    // cursor twice per action and left every later tile
                    // showing "frame pending…". It has its own panel above.
                    // (`includes`, not startsWith: the event mirror joins
                    // adjacent texts, so the label often arrives glued after
                    // a "[frame#N]" marker. Match the STABLE "IMAGE 2" head
                    // only — the tail is the canonical map legend and its
                    // wording legitimately evolves; pinning the old tail
                    // ("— accumulated…") silently re-counted every map as a
                    // camera viewpoint when §20.4 reworded the legend, and
                    // every later tile went back to "frame pending…".)
                    if (prev != null && prev.includes("IMAGE 2"))
                      continue;
                    labels.push(
                      prev != null && prev !== IMG && !prev.trim().startsWith("{")
                        ? (prev.includes("IMAGE 1 —") ? "current view" : prev)
                        : "",
                    );
                  }
                  nViews = labels.length;
                  // Pair by ENV STEP, not by a running cursor: the result's
                  // own steps_taken_total names the obs_*_stepNNN group that
                  // was written with it. The cursor drifted whenever an
                  // internal look wrote a group no result ever showed, and
                  // every later line rendered someone else's frame or
                  // "frame pending…".
                  let stepKey: string | null = null;
                  for (const t of res) {
                    const m = String(t).match(/"steps_taken_total":\s*(\d+)/);
                    if (m) stepKey = m[1].padStart(3, "0");
                  }
                  const byStep = stepKey != null
                    ? frames.filter(
                        (f) => f.includes(`_step${stepKey}`) && !f.includes("_depth"))
                    : [];
                  if (nViews === 1 && byStep.length) {
                    const rgb = byStep[byStep.length - 1];
                    const gm = rgb.match(/^obs_(\d+)_/);
                    const g = gm != null ? groups[groupAt[gm[1]]] : undefined;
                    tiles.push({ url: rgb, label: labels[0] || null });
                    if (g?.depth) tiles.push({ url: g.depth, label: "depth" });
                  } else {
                    for (let v = 0; v < nViews; v++) {
                      const g = groups[groupCursor + v];
                      tiles.push({ url: g?.rgb ?? null, label: labels[v] || null });
                      if (g?.depth) tiles.push({ url: g.depth, label: "depth" });
                    }
                    groupCursor += nViews;
                  }
                }
              }
              // Thinking content is withheld upstream (signature-only blocks,
              // always 0 chars) — rendering them is pure noise.
              if (line.kind === "thinking" && !line.chars) return null;
              // Full input snapshot: the system prompt, first user message, and
              // the entire options object the session ran with. Collapsed by
              // default — long and constant, but it's the INPUT side of the log
              // (the events below are the outputs).
              if (line.kind === "session_inputs") {
                const sp = String(line.system_prompt ?? "");
                const fp = line.first_prompt != null ? String(line.first_prompt) : null;
                const opts = line.options;
                return (
                  <details
                    key={i}
                    className="my-1 rounded border border-gray-800 bg-gray-950/50"
                  >
                    <summary className="cursor-pointer select-none px-2 py-1 text-gray-400">
                      <span className="mr-1 text-gray-600">{line.t.toFixed(1)}s</span>
                      <span className="mr-1">📜</span>
                      session inputs · model {String(line.model ?? "default")} ·{" "}
                      {line.skill ? `skill: ${String(line.skill)}` : "no skill"} ·{" "}
                      {sp.length} chars
                    </summary>
                    <div className="border-t border-gray-800 px-3 py-2">
                      <div className="mb-0.5 text-gray-500">system prompt</div>
                      <pre className="mb-2 whitespace-pre-wrap break-words text-gray-300">
                        {sp}
                      </pre>
                      {fp != null && (
                        <>
                          <div className="mb-0.5 text-gray-500">first user message</div>
                          <pre className="mb-2 whitespace-pre-wrap break-words text-gray-300">
                            {fp}
                          </pre>
                        </>
                      )}
                      {line.tool_schemas != null && (
                        <>
                          <div className="mb-0.5 text-gray-500">tool schemas</div>
                          <pre className="mb-2 whitespace-pre-wrap break-words text-gray-400">
                            {JSON.stringify(line.tool_schemas, null, 2)}
                          </pre>
                        </>
                      )}
                      {(
                        [
                          ["options", opts],
                          ["agent config", line.agent_config],
                          ["model config", line.model_config],
                          ["environment config", line.environment_config],
                        ] as const
                      ).map(
                        ([label, val]) =>
                          val != null && (
                            <div key={label}>
                              <div className="mb-0.5 text-gray-500">{label}</div>
                              <pre className="mb-2 whitespace-pre-wrap break-words text-gray-400">
                                {JSON.stringify(val, null, 2)}
                              </pre>
                            </div>
                          ),
                      )}
                    </div>
                  </details>
                );
              }
              // Final SDK ResultMessage — session cost/turns/stop_reason, as a
              // collapsible line (full object inside).
              if (line.kind === "result") {
                const r = (line.result ?? {}) as Record<string, unknown>;
                return (
                  <details
                    key={i}
                    className="my-1 rounded border border-gray-800 bg-gray-950/50"
                  >
                    <summary className="cursor-pointer select-none px-2 py-1 text-gray-400">
                      <span className="mr-1 text-gray-600">{line.t.toFixed(1)}s</span>
                      <span className="mr-1">🧾</span>
                      result · {String(r.num_turns ?? "?")} turns
                      {r.total_cost_usd != null &&
                        ` · $${Number(r.total_cost_usd).toFixed(4)}`}
                      {r.is_error ? " · ERROR" : ""}
                      {r.stop_reason != null && ` · ${String(r.stop_reason)}`}
                    </summary>
                    <pre className="whitespace-pre-wrap break-words border-t border-gray-800 px-3 py-2 text-gray-400">
                      {JSON.stringify(r, null, 2)}
                    </pre>
                  </details>
                );
              }
              // End-of-episode metrics — fenced off by a divider so the run
              // result reads clearly apart from the trajectory above it.
              if (line.kind === "episode_metrics") {
                const m = (line.metrics ?? {}) as Record<string, number>;
                const fmt = (v: number) =>
                  typeof v === "number" && !Number.isInteger(v)
                    ? v.toFixed(3)
                    : String(v);
                const order = [
                  "success",
                  "spl",
                  "oracle_success",
                  "ndtw",
                  "distance_to_goal",
                  "path_length",
                  "steps_taken",
                ];
                const keys = [
                  ...order.filter((k) => k in m),
                  ...Object.keys(m).filter((k) => !order.includes(k)),
                ];
                return (
                  <div key={i} className="my-2 border-t-2 border-gray-700 pt-2">
                    <div className="mb-1 text-gray-400">
                      <span className="mr-1 text-gray-600">
                        {line.t.toFixed(1)}s
                      </span>
                      <span className="mr-1">🏁</span>
                      run metrics
                    </div>
                    <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                      {keys.map((k) => (
                        <span key={k}>
                          <span className="text-gray-500">{k}</span>{" "}
                          <span className="text-gray-100">{fmt(m[k])}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                );
              }
              const { icon, text, dim } = lineText(line);
              const showFrames =
                tiles.length > 0 && run != null && shownEpisode != null;
              return (
                <div
                  key={i}
                  className={clsx(
                    "whitespace-pre-wrap py-0.5",
                    dim ? "text-gray-500" : "text-gray-200",
                  )}
                >
                  <span className="mr-1 text-gray-600">{line.t.toFixed(1)}s</span>
                  <span className="mr-1">{icon}</span>
                  {text}
                  {showFrames && (
                    // one viewpoint → native-res tiles (observe: RGB + depth);
                    // many → smaller labeled tiles (a look_around panorama). The
                    // tiles, their order, and their labels all come from the
                    // on-disk obs groups paired to the tool result above.
                    // items-end so a captioned tile (depth) and an uncaptioned
                    // one (RGB) still line up their images along the bottom edge.
                    <div className="mt-1 flex flex-wrap items-end gap-1">
                      {tiles.map((tile, k) => {
                        const frame = tile.url;
                        // height is the only fixed dimension: frames keep
                        // their native aspect ratio (egocentric obs are
                        // square; wp panorama strips are ~4:1 and must not
                        // be squashed). max-w-full + object-contain degrade
                        // gracefully when a strip outgrows the log pane.
                        const heightCls = nViews <= 1 ? "h-56" : "h-32";
                        const label = tile.label;
                        return (
                          <div key={k} className="flex flex-col items-center">
                            {label && (
                              <span className="text-[10px] text-gray-600">
                                {label}
                              </span>
                            )}
                            {frame ? (
                              <img
                                src={frameUrl(frame)}
                                alt={frame}
                                title={`${frame} — click to enlarge`}
                                onClick={() => setZoomFrame(frame)}
                                className={clsx(
                                  "block w-auto max-w-full object-contain cursor-zoom-in rounded border border-gray-800",
                                  heightCls,
                                )}
                              />
                            ) : (
                              <div
                                className={clsx(
                                  "flex items-center justify-center rounded border border-dashed border-gray-800 text-gray-600",
                                  heightCls,
                                  nViews <= 1 ? "w-56" : "w-32",
                                )}
                              >
                                frame pending…
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            });
          })()}
          </div>
        </div>
      </div>

      {/* ── lightbox ── */}
      {zoomFrame && run != null && shownEpisode != null && (
        <div
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/80"
          onClick={() => setZoomFrame(null)}
        >
          <img
            src={frameUrl(zoomFrame)}
            alt={zoomFrame}
            className="max-h-[90vh] max-w-[90vw] rounded border border-gray-700"
          />
        </div>
      )}
    </div>
  );
}
