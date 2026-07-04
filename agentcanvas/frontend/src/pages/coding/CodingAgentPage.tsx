import { useEffect, useRef, useState } from "react";
import { Bot, Play, Square } from "lucide-react";
import clsx from "clsx";

// Coding-Agent Monitor — control panel + live text/image logs for agent20
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
    default:
      return { icon: "·", text: JSON.stringify(line), dim: true };
  }
}

export default function CodingAgentPage() {
  // control panel form
  const [episodes, setEpisodes] = useState("0-9");
  const [split, setSplit] = useState("rand100");
  const [maxTurns, setMaxTurns] = useState(80);
  const [model, setModel] = useState("");
  const [startError, setStartError] = useState<string | null>(null);

  // live state
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [viewEpisode, setViewEpisode] = useState<number | null>(null); // null = follow active
  const [lines, setLines] = useState<LogLine[]>([]);
  const [frames, setFrames] = useState<string[]>([]);
  const [zoomFrame, setZoomFrame] = useState<string | null>(null); // lightbox overlay

  const offsetRef = useRef(0);
  // Reset key is run+episode: a new run reuses episode indices (a fresh run's
  // episode 0 must not inherit the old run's log offset).
  const shownKeyRef = useRef<string | null>(null);
  const runRef = useRef<string | null>(null);
  const logBoxRef = useRef<HTMLDivElement | null>(null);

  const running = status?.state === "running" || status?.state === "starting";
  const shownEpisode =
    viewEpisode ??
    status?.active_episode ??
    (status?.started_episodes?.length
      ? status.started_episodes[status.started_episodes.length - 1]
      : null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const st: RunStatus = await (await fetch("/api/coding-agent/status")).json();
        if (cancelled) return;
        setStatus(st);

        if (runRef.current !== st.run_name) {
          runRef.current = st.run_name;
          setViewEpisode(null);
          setZoomFrame(null);
        }

        const run = st.run_name;
        const ep =
          viewEpisode ??
          st.active_episode ??
          (st.started_episodes.length
            ? st.started_episodes[st.started_episodes.length - 1]
            : null);
        if (run == null || ep == null) return;

        const shownKey = `${run}:${ep}`;
        if (shownKeyRef.current !== shownKey) {
          shownKeyRef.current = shownKey;
          offsetRef.current = 0;
          setLines([]);
          setFrames([]);
          setZoomFrame(null);
        }

        const [logRes, framesRes] = await Promise.all([
          fetch(
            `/api/coding-agent/runs/${run}/episode/${ep}/textlog?offset=${offsetRef.current}`,
          ),
          fetch(`/api/coding-agent/runs/${run}/episode/${ep}/frames`),
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
  }, [viewEpisode]);

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

  const run = status?.run_name;
  const frameUrl = (name: string) =>
    `/api/coding-agent/runs/${run}/episode/${shownEpisode}/frame/${name}`;
  const epSummary = (i: number) => status?.episodes.find((e) => e.index === i);

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

      {/* ── episode selector ── */}
      {status && status.started_episodes.length > 0 && (
        <div className="flex flex-wrap items-center gap-1 text-xs">
          <span className="mr-1 text-gray-500">episode:</span>
          {status.started_episodes.map((i) => {
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
          {viewEpisode != null && (
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
        <div className="border-b border-gray-800 px-3 py-1.5 text-xs font-semibold text-gray-400">
          Live Log {shownEpisode != null && `— episode ${shownEpisode}`}
        </div>
        <div ref={logBoxRef} className="min-h-0 flex-1 overflow-y-auto p-2 font-mono text-xs">
          {lines.length === 0 && (
            <div className="p-4 text-gray-600">no events yet</div>
          )}
          {(() => {
            // The k-th observe call maps to the k-th dumped frame (sorted
            // obs_*.png). Counted over ALL lines in order, so the pairing is
            // stable regardless of which kinds get filtered from display.
            let obsSeen = 0;
            return lines.map((line, i) => {
              const isObserve =
                line.kind === "tool_use" &&
                String(line.name ?? "").endsWith("observe");
              const frame = isObserve ? (frames[obsSeen++] ?? null) : null;
              // Thinking content is withheld upstream (signature-only blocks,
              // always 0 chars) — rendering them is pure noise.
              if (line.kind === "thinking" && !line.chars) return null;
              const { icon, text, dim } = lineText(line);
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
                  {isObserve &&
                    run != null &&
                    shownEpisode != null &&
                    (frame ? (
                      // Fixed 224px (native resolution, 1:1) so late-loading
                      // images never shift the scroll position.
                      <img
                        src={frameUrl(frame)}
                        alt={frame}
                        title={`${frame} — click to enlarge`}
                        onClick={() => setZoomFrame(frame)}
                        className="mt-1 block h-56 w-56 cursor-zoom-in rounded border border-gray-800"
                      />
                    ) : (
                      <div className="mt-1 flex h-56 w-56 items-center justify-center rounded border border-dashed border-gray-800 text-gray-600">
                        frame pending…
                      </div>
                    ))}
                </div>
              );
            });
          })()}
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
