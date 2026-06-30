import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import OutputDrawer from "./OutputDrawer";
import { useErrorStore } from "../../errorStore";

const HEIGHT_KEY = "agentcanvas.bottomPanelHeight";
const COLLAPSED_KEY = "agentcanvas.bottomPanelCollapsed";
const MIN_HEIGHT = 80;
const DEFAULT_HEIGHT = 200;
const COLLAPSED_HEIGHT = 28;

function readStoredHeight(): number {
  try {
    const raw = localStorage.getItem(HEIGHT_KEY);
    if (!raw) return DEFAULT_HEIGHT;
    const n = Number(raw);
    return Number.isFinite(n) && n >= MIN_HEIGHT ? n : DEFAULT_HEIGHT;
  } catch {
    return DEFAULT_HEIGHT;
  }
}

function readStoredCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function clampHeight(h: number): number {
  const max = Math.max(MIN_HEIGHT, window.innerHeight - 200);
  return Math.min(max, Math.max(MIN_HEIGHT, h));
}

export default function ResizableBottomPanel() {
  const [height, setHeight] = useState<number>(() =>
    clampHeight(readStoredHeight()),
  );
  const [collapsed, setCollapsed] = useState<boolean>(() =>
    readStoredCollapsed(),
  );
  const dragStateRef = useRef<{ startY: number; startHeight: number } | null>(
    null,
  );

  useEffect(() => {
    try {
      localStorage.setItem(HEIGHT_KEY, String(height));
    } catch {
      /* ignore quota / privacy-mode errors */
    }
  }, [height]);

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  useEffect(() => {
    const onResize = () => setHeight((h) => clampHeight(h));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Expand the panel whenever someone requests focus on the Report tab.
  const focusReportTick = useErrorStore((s) => s.focusReportTick);
  useEffect(() => {
    if (focusReportTick > 0) setCollapsed(false);
  }, [focusReportTick]);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (collapsed) return;
      e.preventDefault();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      dragStateRef.current = { startY: e.clientY, startHeight: height };
    },
    [height, collapsed],
  );

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragStateRef.current;
    if (!drag) return;
    const dy = e.clientY - drag.startY;
    setHeight(clampHeight(drag.startHeight - dy));
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStateRef.current) return;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    dragStateRef.current = null;
  }, []);

  const outerHeight = collapsed ? COLLAPSED_HEIGHT : height;

  return (
    <div
      style={{
        flex: `0 0 ${outerHeight}px`,
        display: "flex",
        flexDirection: "column",
        minHeight: COLLAPSED_HEIGHT,
        position: "relative",
      }}
    >
      {!collapsed && (
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label="Resize bottom panel"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          className="resize-handle"
          title="Drag to resize"
        >
          <div className="resize-handle-grip" />
        </div>
      )}
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="absolute right-2 z-10 flex h-5 w-5 items-center justify-center rounded text-gray-500 hover:bg-gray-800 hover:text-gray-300"
        style={{ top: collapsed ? 4 : 8 }}
        title={collapsed ? "Expand bottom panel" : "Collapse bottom panel"}
      >
        {collapsed ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {collapsed && (
        <div className="flex h-full items-center border-b border-gray-800 bg-gray-900 px-3 text-[10px] uppercase tracking-wider text-gray-500">
          Output panel (collapsed)
        </div>
      )}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: collapsed ? "none" : "block",
        }}
      >
        <OutputDrawer />
      </div>
    </div>
  );
}
