import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Zap,
} from "lucide-react";
import { evalApi } from "../eval/evalApi";
import type { EvalRunSummary } from "../eval/types";
import {
  replayApi,
  type ReplayEpisode,
  type ReplayEpisodesResponse,
} from "./replayApi";

const PLAY_INTERVAL_MS = 800;
// Smooth-mode default: 10 frames per step interval = 100 ms each at 10 fps.
const SMOOTH_FPS_DEFAULT = 10;
const SMOOTH_FRAMES_PER_STEP_DEFAULT = 10;

export default function ReplayPage() {
  const [runs, setRuns] = useState<EvalRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<EvalRunSummary | null>(null);
  const [episodes, setEpisodes] = useState<ReplayEpisodesResponse | null>(null);
  const [episodesError, setEpisodesError] = useState<string | null>(null);
  const [selectedEpIdx, setSelectedEpIdx] = useState<number | null>(null);
  const [episode, setEpisode] = useState<ReplayEpisode | null>(null);
  const [episodeError, setEpisodeError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [smoothMode, setSmoothMode] = useState(false);
  const [smoothFps, setSmoothFps] = useState(SMOOTH_FPS_DEFAULT);
  const [smoothFrac, setSmoothFrac] = useState(0); // 0 = at currentStep, advances toward 1
  const playTimer = useRef<number | null>(null);

  // Load runs on mount.
  useEffect(() => {
    loadRuns();
  }, []);

  function loadRuns() {
    setRunsError(null);
    evalApi
      .listRuns()
      .then(({ runs: r }) => setRuns(r || []))
      .catch((e: unknown) => setRunsError(String(e)));
  }

  // Run selection — reset everything synchronously in the click handler
  // (not via useEffect cascade) so the episode-fetch effect never sees a
  // stale ``selectedEpIdx`` against a freshly-changed run. Concretely:
  // batched useEffects in the same render fire with the values from THIS
  // render, so an "old ep_idx + new run_id" call could 404.
  function selectRun(r: EvalRunSummary) {
    setSelectedRun(r);
    setEpisodes(null);
    setEpisodesError(null);
    setSelectedEpIdx(null);
    setEpisode(null);
    setEpisodeError(null);
    setCurrentStep(0);
    setIsPlaying(false);
    setSmoothMode(false);
    replayApi
      .listEpisodes(r.run_id)
      .then((d) => setEpisodes(d))
      .catch((e: unknown) => setEpisodesError(String(e)));
  }

  // Load full episode when one is selected. Guard the fetch by checking
  // the run_id at resolution time so stale promises from a previous run
  // (during a fast run-swap) don't overwrite the current episode state.
  useEffect(() => {
    if (!selectedRun || selectedEpIdx === null) return;
    const runId = selectedRun.run_id;
    const epIdx = selectedEpIdx;
    setEpisode(null);
    setEpisodeError(null);
    setCurrentStep(0);
    setIsPlaying(false);
    replayApi
      .getEpisode(runId, epIdx)
      .then((d) => {
        if (selectedRun?.run_id !== runId || selectedEpIdx !== epIdx) return;
        setEpisode(d);
      })
      .catch((e: unknown) => {
        if (selectedRun?.run_id !== runId || selectedEpIdx !== epIdx) return;
        setEpisodeError(String(e));
      });
  }, [selectedRun, selectedEpIdx]);

  // Reset smoothFrac whenever the user navigates manually or smooth mode toggles.
  useEffect(() => {
    setSmoothFrac(0);
  }, [currentStep, smoothMode, episode]);

  // Play loop — step mode (v0) advances currentStep at fixed interval.
  useEffect(() => {
    if (!isPlaying || !episode || smoothMode) return;
    playTimer.current = window.setInterval(() => {
      setCurrentStep((s) => {
        if (s >= episode.steps.length - 1) {
          setIsPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, PLAY_INTERVAL_MS);
    return () => {
      if (playTimer.current !== null) {
        clearInterval(playTimer.current);
        playTimer.current = null;
      }
    };
  }, [isPlaying, episode, smoothMode]);

  // Play loop — smooth mode (v1) advances smoothFrac in 1/N increments;
  // when it overflows past 1, advance to the next step.
  useEffect(() => {
    if (!isPlaying || !episode || !smoothMode) return;
    const tickMs = Math.max(20, Math.round(1000 / smoothFps));
    const step = 1 / SMOOTH_FRAMES_PER_STEP_DEFAULT;
    playTimer.current = window.setInterval(() => {
      setSmoothFrac((f) => {
        const next = f + step;
        if (next >= 1) {
          setCurrentStep((s) => {
            if (s >= episode.steps.length - 2) {
              setIsPlaying(false);
              return s;
            }
            return s + 1;
          });
          return 0;
        }
        return next;
      });
    }, tickMs);
    return () => {
      if (playTimer.current !== null) {
        clearInterval(playTimer.current);
        playTimer.current = null;
      }
    };
  }, [isPlaying, episode, smoothMode, smoothFps]);

  const step = episode?.steps[currentStep] ?? null;
  const isLastStep = episode ? currentStep >= episode.steps.length - 1 : false;
  // In smooth mode with smoothFrac > 0 and not at last step, request an
  // interpolated frame from the backend; otherwise show the saved keyframe.
  const useSmoothFrame =
    smoothMode && !!episode && !isLastStep && smoothFrac > 0 && !!selectedRun;
  const frameSrc = useSmoothFrame
    ? replayApi.smoothFrameUrl(
        selectedRun!.run_id,
        episode!.episode_index,
        currentStep,
        smoothFrac,
      )
    : episode && step?.frame_url
      ? replayApi.assetUrl(episode.execution_id, step.frame_url)
      : "";

  return (
    <div className="flex h-full bg-gray-950 text-gray-100">
      {/* Left: Run + Episode pickers */}
      <div className="flex w-72 flex-col border-r border-gray-800">
        <div className="flex items-center justify-between border-b border-gray-800 p-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
            Runs
          </h2>
          <button
            onClick={loadRuns}
            className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-200"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {runsError && (
            <div className="p-3 text-xs text-red-400">{runsError}</div>
          )}
          {runs.map((r) => (
            <button
              key={r.run_id}
              onClick={() => selectRun(r)}
              className={clsx(
                "block w-full border-b border-gray-800 px-3 py-2 text-left text-xs hover:bg-gray-800",
                selectedRun?.run_id === r.run_id && "bg-gray-800",
              )}
            >
              <div className="font-mono text-gray-300">{r.run_id}</div>
              <div className="mt-0.5 text-gray-500">
                {r.graph_name} · {r.env_nodeset} · {r.total_episodes} eps
              </div>
            </button>
          ))}
          {!runs.length && !runsError && (
            <div className="p-3 text-xs text-gray-500">No runs yet.</div>
          )}
        </div>

        <div className="border-t border-gray-800 p-3">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-gray-400">
            Episodes
          </h2>
          {episodesError && (
            <div className="text-xs text-red-400">{episodesError}</div>
          )}
          {episodes && (
            <div className="max-h-64 overflow-y-auto">
              {episodes.episodes.map((ep) => {
                const success = ep.metrics?.success;
                return (
                  <button
                    key={ep.episode_index}
                    onClick={() => setSelectedEpIdx(ep.episode_index)}
                    className={clsx(
                      "block w-full rounded px-2 py-1.5 text-left text-xs hover:bg-gray-800",
                      selectedEpIdx === ep.episode_index && "bg-gray-800",
                    )}
                  >
                    <span className="font-mono">ep {ep.episode_index}</span>
                    <span className="ml-2 text-gray-500">
                      {ep.step_count} steps
                    </span>
                    {success !== undefined && (
                      <span
                        className={clsx(
                          "ml-2",
                          success > 0 ? "text-green-400" : "text-red-400",
                        )}
                      >
                        {success > 0 ? "✓" : "✗"}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
          {!episodes && !episodesError && selectedRun && (
            <div className="text-xs text-gray-500">Loading…</div>
          )}
        </div>
      </div>

      {/* Center: Frame viewer + step navigator */}
      <div className="flex flex-1 flex-col">
        <div className="flex flex-1 items-center justify-center bg-black p-4">
          {episodeError && (
            <div className="text-sm text-red-400">{episodeError}</div>
          )}
          {!episode && !episodeError && (
            <div className="text-sm text-gray-500">
              {selectedEpIdx !== null
                ? "Loading episode…"
                : "Pick a run, then an episode."}
            </div>
          )}
          {frameSrc && (
            <img
              src={frameSrc}
              alt={`step ${currentStep}`}
              className="max-h-full max-w-full object-contain"
            />
          )}
          {episode && !frameSrc && (
            <div className="text-sm text-gray-500">
              No frame asset for this step.
            </div>
          )}
        </div>

        {episode && (
          <div className="flex items-center gap-3 border-t border-gray-800 bg-gray-900 px-4 py-2">
            <button
              onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
              disabled={currentStep === 0}
              className="rounded p-1 text-gray-300 hover:bg-gray-800 disabled:opacity-30"
              title="Prev"
            >
              <ChevronLeft size={20} />
            </button>
            <button
              onClick={() => setIsPlaying((p) => !p)}
              className="rounded bg-blue-600 p-1 text-white hover:bg-blue-500"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button
              onClick={() =>
                setCurrentStep((s) => Math.min(episode.steps.length - 1, s + 1))
              }
              disabled={currentStep >= episode.steps.length - 1}
              className="rounded p-1 text-gray-300 hover:bg-gray-800 disabled:opacity-30"
              title="Next"
            >
              <ChevronRight size={20} />
            </button>
            <span className="font-mono text-xs text-gray-400">
              step {currentStep + 1} / {episode.steps.length}
              {smoothMode && smoothFrac > 0 && (
                <span className="ml-1 text-blue-400">
                  +{smoothFrac.toFixed(2)}
                </span>
              )}
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, episode.steps.length - 1)}
              value={currentStep}
              onChange={(e) => {
                setIsPlaying(false);
                setCurrentStep(Number(e.target.value));
              }}
              className="flex-1"
            />
            {episode.supports_smooth && (
              <>
                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setSmoothMode((m) => !m);
                  }}
                  className={clsx(
                    "flex items-center gap-1 rounded px-2 py-1 text-xs",
                    smoothMode
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-300 hover:bg-gray-700",
                  )}
                  title="Smooth mode — render interpolated frames between steps"
                >
                  <Zap size={14} /> Smooth
                </button>
                {smoothMode && (
                  <label className="flex items-center gap-1 text-xs text-gray-400">
                    <span>fps</span>
                    <input
                      type="number"
                      min={1}
                      max={30}
                      value={smoothFps}
                      onChange={(e) =>
                        setSmoothFps(
                          Math.max(
                            1,
                            Math.min(30, Number(e.target.value) || 1),
                          ),
                        )
                      }
                      className="w-12 rounded bg-gray-800 px-1 py-0.5 font-mono text-gray-200"
                    />
                  </label>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* Right: Info sidebar */}
      <div className="w-80 overflow-y-auto border-l border-gray-800 bg-gray-900 p-4">
        {episode ? (
          <>
            <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-gray-400">
              Instruction
            </h2>
            <p className="mb-4 text-sm text-gray-200">{episode.instruction}</p>

            {Object.keys(episode.metrics || {}).length > 0 && (
              <>
                <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-gray-400">
                  Episode metrics
                </h2>
                <div className="mb-4 space-y-0.5 font-mono text-xs">
                  {Object.entries(episode.metrics).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-gray-500">{k}</span>
                      <span className="text-gray-200">
                        {typeof v === "number" ? v.toFixed(3) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {step && (
              <>
                <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-gray-400">
                  Step {step.step_index} info
                </h2>
                <div className="space-y-2 text-xs">
                  {Object.entries(step.info).map(([k, v]) => (
                    <div key={k}>
                      <div className="font-mono text-gray-500">{k}</div>
                      <div className="whitespace-pre-wrap break-words text-gray-200">
                        {String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="text-xs text-gray-500">No episode loaded.</div>
        )}
      </div>
    </div>
  );
}
