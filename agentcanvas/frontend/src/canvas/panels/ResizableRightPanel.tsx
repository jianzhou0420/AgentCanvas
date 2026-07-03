/** Right side panel hosting the Properties view.
 *
 * Tall-and-narrow content (property rows, config sections) fits a side
 * panel better than the bottom drawer. Collapsible to a thin strip;
 * auto-expands when a node is selected so the edit surface is
 * discoverable from a plain node click.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import PropertiesPanel from "./PropertiesPanel";
import { useFlowStore } from "../useFlowStore";

const WIDTH_KEY = "agentcanvas.rightPanelWidth";
const COLLAPSED_KEY = "agentcanvas.rightPanelCollapsed";
const MIN_WIDTH = 220;
const DEFAULT_WIDTH = 300;
const MAX_WIDTH_PAD = 400; // leave at least this much for the canvas
const COLLAPSED_WIDTH = 24;

function readStoredWidth(): number {
  try {
    const raw = localStorage.getItem(WIDTH_KEY);
    if (!raw) return DEFAULT_WIDTH;
    const n = Number(raw);
    return Number.isFinite(n) && n >= MIN_WIDTH ? n : DEFAULT_WIDTH;
  } catch {
    return DEFAULT_WIDTH;
  }
}

function readStoredCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function clampWidth(w: number): number {
  const max = Math.max(MIN_WIDTH, window.innerWidth - MAX_WIDTH_PAD);
  return Math.min(max, Math.max(MIN_WIDTH, w));
}

export default function ResizableRightPanel() {
  const [width, setWidth] = useState<number>(() =>
    clampWidth(readStoredWidth()),
  );
  const [collapsed, setCollapsed] = useState<boolean>(() =>
    readStoredCollapsed(),
  );
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(
    null,
  );

  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      /* ignore quota / privacy-mode errors */
    }
  }, [width]);

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  useEffect(() => {
    const onResize = () => setWidth((w) => clampWidth(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Selecting a node is the natural "show me its properties" gesture —
  // pop the panel open so the edit surface never has to be hunted down.
  const selectedNodeId = useFlowStore((s) => s.selectedNodeId);
  useEffect(() => {
    if (selectedNodeId) setCollapsed(false);
  }, [selectedNodeId]);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (collapsed) return;
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      dragStateRef.current = { startX: e.clientX, startWidth: width };
    },
    [width, collapsed],
  );

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    // Handle sits on the panel's left edge: dragging left grows the panel.
    setWidth(clampWidth(drag.startWidth - dx));
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStateRef.current) return;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    dragStateRef.current = null;
  }, []);

  if (collapsed) {
    return (
      <div
        style={{ width: COLLAPSED_WIDTH, flexShrink: 0 }}
        className="relative flex flex-col items-center border-l border-gray-800 bg-gray-900"
      >
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          className="mt-1 flex h-5 w-5 items-center justify-center rounded text-gray-500 hover:bg-gray-800 hover:text-gray-300"
          title="Expand properties panel"
        >
          <ChevronLeft size={12} />
        </button>
        <span
          className="mt-2 text-[9px] uppercase tracking-wider text-gray-600"
          style={{ writingMode: "vertical-rl" }}
        >
          Properties
        </span>
      </div>
    );
  }

  return (
    <div
      style={{ width: `${width}px`, flexShrink: 0 }}
      className="relative flex flex-col border-l border-gray-800 bg-gray-900"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize properties panel"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        className="resize-handle-vertical absolute left-0 top-0 z-10 h-full"
        title="Drag to resize · double-click to reset"
      />
      <button
        type="button"
        onClick={() => setCollapsed(true)}
        className="absolute right-1 top-1 z-10 flex h-5 w-5 items-center justify-center rounded text-gray-500 hover:bg-gray-800 hover:text-gray-300"
        title="Collapse properties panel"
      >
        <ChevronRight size={12} />
      </button>
      <div className="min-h-0 flex-1">
        <PropertiesPanel />
      </div>
    </div>
  );
}
