import { useEffect, useRef, useState } from "react";
import { Bot, Download, Play, Square } from "lucide-react";
import clsx from "clsx";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import { usePersistentState } from "./usePersistentState";

// Coding-Agent Monitor — control panel + live text/image logs for beta-coding-agent
// runs (vanilla coding agent driving env_habitat through the MCP bridge).
// v1: single worker, one run at a time. Live data is 1 Hz polling against
// /api/coding-agent (the trajectory JSONL is flushed per event backend-side).

const POLL_MS = 1000;

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
  config: { episodes?: string; split?: string; max_turns?: number; model?: string | null };
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
      return { icon: "🔧", text: `${name} ${JSON.stringify(line.input)}`, dim: false };
    }
    case "tool_result": {
      const texts = (line.texts as string[] | undefined) ?? [];
      return { icon: "↩", text: texts.join(" ").slice(0, 300), dim: true };
    }
    // mini-swe-agent event kinds (beta-react-harness runs)
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
    default:
      return { icon: "·", text: JSON.stringify(line), dim: true };
  }
}

export default function CodingAgentPage() {
  // control panel form (persisted: survive a refresh with the same inputs)
  const [episodes, setEpisodes] = usePersistentState("agentcanvas.coding.episodes", "0-9");
  const [split, setSplit] = usePersistentState("agentcanvas.coding.split", "rand100");
  const [maxTurns, setMaxTurns] = usePersistentState("agentcanvas.coding.maxTurns", 80);
  const [model, setModel] = usePersistentState("agentcanvas.coding.model", "");
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

  // log browser (any run under outputs/beta-coding-agent/, CLI-launched included)
  const [mode, setMode] = usePersistentState<"live" | "browse">(
    "agentcanvas.coding.mode",
    "live",
  );
  // which harness's runs to browse: Agent SDK (beta-coding-agent) vs
  // mini-swe-agent (beta-react-harness) vs OpenAI Codex CLI
  // (beta-codex-agent). Live mode is SDK-runner-only.
  const [harness, setHarness] = usePersistentState<"claude-sdk" | "mini-swe" | "codex">(
    "agentcanvas.coding.harness",
    "claude-sdk",
  );
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
        if (run == null || ep == null) return;

        const shownKey = `${run}:${ep}`;
        if (shownKeyRef.current !== shownKey) {
          shownKeyRef.current = shownKey;
          offsetRef.current = 0;
          setLines([]);
          setFrames([]);
          setZoomFrame(null);
        }

        const src = mode === "browse" ? harness : "claude-sdk";
        const [logRes, framesRes] = await Promise.all([
          fetch(
            `/api/coding-agent/runs/${run}/episode/${ep}/textlog?offset=${offsetRef.current}&source=${src}`,
          ),
          fetch(`/api/coding-agent/runs/${run}/episode/${ep}/frames?source=${src}`),
        ]);
        const logData = await logRes.json();
        const framesData = await framesRes.json();
        if (cancelled) return;

        if (logData.lines.length > 0) {
          offsetRef.current = logData.next_offset;
          setLines((prev) => [...prev, ...logData.lines]);
        }
        setFrames(framesData.frames);
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

  const loadRuns = async (source?: "claude-sdk" | "mini-swe" | "codex") => {
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
    if (mode !== "browse" || !browseRun) return;
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
    const res = await fetch("/api/coding-agent/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episodes,
        split,
        max_turns: maxTurns,
        model: model.trim() || null,
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
  const frameUrl = (name: string) =>
    `/api/coding-agent/runs/${run}/episode/${shownEpisode}/frame/${name}?source=${
      mode === "browse" ? harness : "claude-sdk"
    }`;
  const epList = mode === "browse" ? browseEpisodes : (status?.episodes ?? []);
  const epSummary = (i: number) => epList.find((e) => e.index === i);
  const selEpisodes = mode === "browse" ? browseStarted : (status?.started_episodes ?? []);
  const browseInfo = runsList.find((r) => r.name === browseRun);

  // Export the currently shown episode's log as ONE tall PDF page (长图):
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
      // it stays a single 长图 strip rather than a fixed A4 sheet.
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
    <div className="flex h-full flex-col gap-3 overflow-hidden bg-gray-950 p-3 text-gray-200">
      {/* ── control panel ── */}
      <div className="flex flex-wrap items-center gap-3 rounded border border-gray-800 bg-gray-900 px-3 py-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-blue-400">
          <Bot size={18} />
          Coding-Agent Monitor
        </div>
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
          model
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={running}
            placeholder="(account default)"
            className="w-36 rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 text-xs text-gray-200"
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
            const badge =
              s == null ? "…" : s.error ? "⚠" : s.success ? "✅" : "❌";
            return (
              <button
                key={i}
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

      {/* ── unified log (frames embedded inline at their observe calls) ── */}
      <div className="flex min-h-0 flex-1 flex-col rounded border border-gray-800 bg-gray-900">
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
              if (line.kind === "tool_use") {
                const res = resultByToolUse[String(line.id ?? "")];
                if (res) {
                  const labels: string[] = [];
                  for (let j = 0; j < res.length; j++) {
                    if (res[j] !== IMG || res[j - 1] === IMG) continue;
                    const prev = j > 0 ? res[j - 1] : undefined;
                    labels.push(
                      prev != null && prev !== IMG && !prev.trim().startsWith("{")
                        ? prev
                        : "",
                    );
                  }
                  nViews = labels.length;
                  for (let v = 0; v < nViews; v++) {
                    const g = groups[groupCursor + v];
                    tiles.push({ url: g?.rgb ?? null, label: labels[v] || null });
                    if (g?.depth) tiles.push({ url: g.depth, label: "depth" });
                  }
                  groupCursor += nViews;
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
