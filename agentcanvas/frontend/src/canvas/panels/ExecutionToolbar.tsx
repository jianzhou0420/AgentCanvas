/** Execution toolbar — Play/Pause/Stop/Step/Reset controls for the root canvas.
 *
 * In the unified canvas, the root canvas IS the graph. This toolbar provides
 * playback controls at the page level. All environment interaction happens
 * through nodeset nodes wired on the canvas.
 */

import {
  Play,
  Pause,
  Square,
  SkipForward,
  RotateCcw,
  Database,
  Eye,
  Bell,
  StickyNote,
} from "lucide-react";
import clsx from "clsx";
import { useFlowStore } from "../useFlowStore";
import { useStore } from "../../store";
import { useErrorStore } from "../../errorStore";
import { runPipeline } from "../runPipeline";

/** Bell icon + unread badge — opens the bottom-panel Report tab. */
function ReportBell() {
  const unread = useErrorStore((s) => s.unreadCount);
  const focus = useErrorStore((s) => s.focusReport);
  return (
    <button
      onClick={focus}
      className={clsx(
        "relative flex items-center gap-1 rounded px-1.5 py-1 text-[10px] transition",
        unread > 0
          ? "text-red-400 hover:bg-red-500/10"
          : "text-gray-500 hover:bg-gray-800 hover:text-gray-300",
      )}
      title={
        unread > 0 ? `${unread} unread events — open Report` : "Open Report"
      }
    >
      <Bell size={11} />
      {unread > 0 && (
        <span className="rounded-full bg-red-500 px-1 py-0 text-[8px] font-bold text-white">
          {unread > 99 ? "99+" : unread}
        </span>
      )}
    </button>
  );
}

/** Toggle button for showing/hiding state edges on the canvas. */
function StateEdgeToggle() {
  const showStateEdges = useFlowStore((s) => s.showStateEdges);
  const toggleStateEdges = useFlowStore((s) => s.toggleStateEdges);
  return (
    <button
      onClick={toggleStateEdges}
      className={clsx(
        "flex items-center gap-1 rounded px-2 py-1 text-[10px] transition",
        showStateEdges
          ? "bg-violet-600/30 text-violet-300 hover:bg-violet-600/50"
          : "bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-400",
      )}
      title="Toggle state edges"
    >
      <Database size={10} />
      State
    </button>
  );
}

/** Toggle for showing/hiding viewer nodes + edges touching them (render-only). */
function ViewerToggle() {
  const showViewers = useFlowStore((s) => s.showViewers);
  const toggleViewers = useFlowStore((s) => s.toggleViewers);
  return (
    <button
      onClick={toggleViewers}
      className={clsx(
        "flex items-center gap-1 rounded px-2 py-1 text-[10px] transition",
        showViewers
          ? "bg-purple-600/30 text-purple-300 hover:bg-purple-600/50"
          : "bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-400",
      )}
      title="Toggle viewer nodes"
    >
      <Eye size={10} />
      Viewer
    </button>
  );
}

/** Toggle for showing/hiding annotation nodes (note + future sticker/link). */
function AnnotationToggle() {
  const showAnnotations = useFlowStore((s) => s.showAnnotations);
  const toggleAnnotations = useFlowStore((s) => s.toggleAnnotations);
  return (
    <button
      onClick={toggleAnnotations}
      className={clsx(
        "flex items-center gap-1 rounded px-2 py-1 text-[10px] transition",
        showAnnotations
          ? "bg-amber-600/30 text-amber-300 hover:bg-amber-600/50"
          : "bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-400",
      )}
      title="Toggle annotation nodes (notes etc.)"
    >
      <StickyNote size={10} />
      Annotation
    </button>
  );
}

export default function ExecutionToolbar() {
  const navStatus = useStore((s) => s.navStatus);
  const currentStep = useStore((s) => s.navCurrentStep);
  const stepDelay = useStore((s) => s.navStepDelay);
  const setNavStepDelay = useStore((s) => s.setNavStepDelay);
  const navPause = useStore((s) => s.navPause);
  const navStop = useStore((s) => s.navStop);
  const stepBudget = useFlowStore(
    (s) => s.tabs[s.activeTabId]?.step_budget ?? 500,
  );
  const setStepBudget = useFlowStore((s) => s.setStepBudget);

  // Playback state machine
  const canPlay =
    navStatus === "idle" || navStatus === "paused" || navStatus === "done";
  const canPause = navStatus === "running";
  const canStop = navStatus === "running" || navStatus === "paused";
  const canStep = navStatus === "idle" || navStatus === "paused";
  const canReset =
    navStatus === "done" || navStatus === "paused" || navStatus === "error";

  const handlePlay = async () => {
    try {
      // Env episode pre-flight (when an env nodeset is loaded) lives on the
      // EnvPanel's Play button — that path calls the env panel's
      // on_action("play") which does the env reset before runPipeline().
      // This toolbar Play is the bare-graph fallback for graphs without
      // an env panel-aware nodeset.
      const graph = useFlowStore.getState().getGraphForExecution();
      if (!graph) return;
      const executionId = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      useFlowStore.getState().startExecution(executionId);
      await runPipeline(graph, executionId);
    } catch (err) {
      console.error("Pipeline error:", err);
    }
  };

  const handleReset = async () => {
    await navStop();
    // Reset clears execution state — the graph can be re-run with Play
    useStore.setState({
      navCurrentStep: 0,
      navStatus: "idle",
      navMetrics: null,
      navSteps: [],
      navLLMSteps: [],
    });
    // Clear output viewer data so they show fresh state (use tab update pattern
    // so both tab.nodeOutputs and the projected nodeOutputs stay in sync)
    useFlowStore.setState((s) => {
      const tab = s.tabs[s.activeTabId];
      if (!tab) return {};
      const updated = { ...tab, nodeOutputs: {}, activeExecutions: {} };
      return {
        tabs: { ...s.tabs, [s.activeTabId]: updated },
        nodeOutputs: {},
        activeExecutions: {},
      };
    });
  };

  return (
    <div className="flex items-center gap-2 border-b border-gray-800 bg-gray-900/95 px-3 py-1.5">
      {/* Playback buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={handlePlay}
          disabled={!canPlay}
          className={clsx(
            "flex items-center justify-center rounded px-2 py-1 text-xs",
            canPlay
              ? "bg-green-600 text-white hover:bg-green-500"
              : "bg-gray-800 text-gray-600",
          )}
          title="Play"
        >
          <Play size={12} />
        </button>
        <button
          onClick={navPause}
          disabled={!canPause}
          className={clsx(
            "flex items-center justify-center rounded px-2 py-1 text-xs",
            canPause
              ? "bg-yellow-600 text-white hover:bg-yellow-500"
              : "bg-gray-800 text-gray-600",
          )}
          title="Pause"
        >
          <Pause size={12} />
        </button>
        <button
          onClick={() => {}}
          disabled={!canStep}
          className={clsx(
            "flex items-center justify-center rounded px-2 py-1 text-xs",
            canStep
              ? "bg-blue-600 text-white hover:bg-blue-500"
              : "bg-gray-800 text-gray-600",
          )}
          title="Step"
        >
          <SkipForward size={12} />
        </button>
        <button
          onClick={navStop}
          disabled={!canStop}
          className={clsx(
            "flex items-center justify-center rounded px-2 py-1 text-xs",
            canStop
              ? "bg-red-600 text-white hover:bg-red-500"
              : "bg-gray-800 text-gray-600",
          )}
          title="Stop"
        >
          <Square size={12} />
        </button>
        <button
          onClick={handleReset}
          disabled={!canReset}
          className={clsx(
            "flex items-center justify-center rounded px-2 py-1 text-xs",
            canReset
              ? "bg-orange-600 text-white hover:bg-orange-500"
              : "bg-gray-800 text-gray-600",
          )}
          title="Reset"
        >
          <RotateCcw size={12} />
        </button>
      </div>

      {/* Step delay slider */}
      <div className="flex items-center gap-1">
        <input
          type="range"
          min={0}
          max={2000}
          step={50}
          value={stepDelay}
          onChange={(e) => setNavStepDelay(Number(e.target.value))}
          className="w-20"
        />
        <span className="w-12 text-right font-mono text-[10px] text-gray-500">
          {stepDelay}ms
        </span>
      </div>

      {/* Max steps */}
      <div className="mx-1 h-4 border-l border-gray-700" />
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-gray-500">Max</span>
        <input
          type="number"
          min={1}
          max={9999}
          step={1}
          value={stepBudget}
          onChange={(e) =>
            setStepBudget(Math.max(1, Number(e.target.value) || 500))
          }
          className="w-14 rounded bg-gray-800 px-1 py-0.5 text-center font-mono text-[10px] text-gray-300 outline-none focus:ring-1 focus:ring-blue-500"
          title="Step budget (per-episode iteration cap) for graph executor"
        />
      </div>

      {/* State edge toggle */}
      <div className="mx-1 h-4 border-l border-gray-700" />
      <StateEdgeToggle />
      <ViewerToggle />
      <AnnotationToggle />

      {/* Status */}
      <span className="text-[10px] text-gray-600">Step {currentStep}</span>
      <span
        className={clsx(
          "text-[10px] font-medium",
          navStatus === "running" && "text-green-400",
          navStatus === "paused" && "text-yellow-400",
          navStatus === "done" && "text-blue-400",
          navStatus === "error" && "text-red-400",
          navStatus === "idle" && "text-gray-600",
        )}
      >
        {navStatus}
      </span>
      <div className="mx-1 h-4 border-l border-gray-700" />
      <ReportBell />
    </div>
  );
}
