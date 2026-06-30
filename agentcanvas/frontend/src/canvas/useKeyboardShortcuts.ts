/** Global keyboard shortcuts for the canvas.
 *
 * Space  — Play if idle/paused/done, else Pause if running.
 * F      — Fit-view on the active ReactFlow instance (exposed via window.__rf).
 * ?      — Toggle a shortcut cheatsheet (caller owns the modal state).
 *
 * Ignores events whose target is an input/textarea/contenteditable element.
 */

import { useEffect } from "react";
import { useFlowStore } from "./useFlowStore";
import { useStore } from "../store";
import { runPipeline } from "./runPipeline";

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return false;
}

async function triggerPlay() {
  const graph = useFlowStore.getState().getGraphForExecution();
  if (!graph) return;
  const executionId = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  useFlowStore.getState().startExecution(executionId);
  try {
    await runPipeline(graph, executionId);
  } catch (err) {
    console.error("Pipeline error:", err);
  }
}

interface Options {
  onToggleCheatsheet: () => void;
}

export function useKeyboardShortcuts({ onToggleCheatsheet }: Options) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;

      if (e.code === "Space") {
        e.preventDefault();
        const status = useStore.getState().navStatus;
        if (status === "running") {
          useStore.getState().navPause();
        } else if (
          status === "idle" ||
          status === "paused" ||
          status === "done"
        ) {
          void triggerPlay();
        }
        return;
      }

      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        const rf = (
          window as unknown as {
            __rf?: { fitView: (o?: { padding?: number }) => void };
          }
        ).__rf;
        rf?.fitView({ padding: 0.2 });
        return;
      }

      if (e.key === "?") {
        e.preventDefault();
        onToggleCheatsheet();
        return;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onToggleCheatsheet]);
}
