/** Source tab (bottom panel) — scoped per-node source editing.
 *
 * Shows only the selected node's slice of its nodeset file: module-level
 * globals, the functions its class transitively references, and the class
 * itself — one stacked editor per segment. Save splices the segments back
 * by line range; the backend syntax-checks the whole file before writing,
 * then the nodeset watcher hot-reloads ("Saved" → "Reloaded ✓" via the
 * components_changed broadcast, mirrored in useSourceEditorStore).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "../../api";
import type { ScopedSource } from "../../api";
import { useFlowStore } from "../useFlowStore";
import { useSourceEditorStore } from "../sourceEditorStore";
import CodeMirrorEditor from "./CodeMirrorEditor";

type SaveState = "idle" | "saving" | "saved" | "deferred" | "reloaded";

const KIND_STYLE: Record<string, string> = {
  globals: "bg-purple-900/50 text-purple-300",
  function: "bg-sky-900/50 text-sky-300",
  class: "bg-emerald-900/50 text-emerald-300",
};

export default function SourcePanel() {
  const selectedNodeId = useFlowStore((s) => s.selectedNodeId);
  const visibleNodes = useFlowStore((s) => s.visibleNodes);
  const node = visibleNodes.find((n) => n.id === selectedNodeId);

  if (!node) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-xs text-gray-600">Select a node to edit its source</span>
      </div>
    );
  }
  if (!node.type?.includes("__")) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-xs text-gray-600">
          Built-in node — source not editable
        </span>
      </div>
    );
  }
  // Keyed remount: all editor state resets when another node is selected.
  return <ScopedEditor key={node.type} nodeType={node.type} />;
}

function ScopedEditor({ nodeType }: { nodeType: string }) {
  const nodesetName = nodeType.split("__")[0];
  const changeSeq = useSourceEditorStore((s) => s.changeSeq);

  const [src, setSrc] = useState<ScopedSource | null>(null);
  const [texts, setTexts] = useState<string[]>([]);
  const [loadSeq, setLoadSeq] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [stale, setStale] = useState(false);
  const [restarting, setRestarting] = useState(false);

  const dirty =
    src !== null && texts.some((t, i) => t !== src.segments[i]?.text);

  const applyScoped = useCallback((data: ScopedSource) => {
    setSrc(data);
    setTexts(data.segments.map((s) => s.text));
    setLoadSeq((n) => n + 1);
    setConflict(false);
    setSaveError(null);
  }, []);

  const load = useCallback(async () => {
    setLoadError(null);
    try {
      applyScoped(await api.getScopedSource(nodesetName, nodeType));
      setSaveState("idle");
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  }, [nodesetName, nodeType, applyScoped]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = useCallback(async () => {
    if (!src || !dirty || saveState === "saving") return;
    setSaveState("saving");
    setSaveError(null);
    setConflict(false);
    const res = await api.saveScopedSource(nodesetName, {
      file: src.file,
      node_type: nodeType,
      base_mtime_ns: src.mtime_ns,
      segments: src.segments.map((s, i) => ({
        start_line: s.start_line,
        end_line: s.end_line,
        text: texts[i] ?? s.text,
      })),
    });
    if (res.ok) {
      if (res.segments) {
        applyScoped({ ...src, mtime_ns: res.mtime_ns, segments: res.segments });
      }
      if (res.stale) setStale(true);
      setSaveState(res.run_active ? "deferred" : "saved");
    } else if (res.kind === "syntax") {
      setSaveState("idle");
      setSaveError(`Syntax error at line ${res.line ?? "?"}: ${res.msg}`);
    } else if (res.kind === "conflict") {
      setSaveState("idle");
      setConflict(true);
    } else {
      setSaveState("idle");
      setSaveError(res.message);
    }
  }, [src, dirty, saveState, texts, nodesetName, nodeType, applyScoped]);

  // Watcher broadcast: flip Saved → Reloaded ✓ / raise the stale banner.
  const changeSeqSeenRef = useRef(changeSeq);
  useEffect(() => {
    if (changeSeq === changeSeqSeenRef.current) return;
    changeSeqSeenRef.current = changeSeq;
    const { lastReloaded, lastStale } = useSourceEditorStore.getState();
    if (lastReloaded.includes(nodesetName)) {
      setSaveState((s) => (s === "saved" || s === "deferred" ? "reloaded" : s));
    }
    if (lastStale.includes(nodesetName)) setStale(true);
  }, [changeSeq, nodesetName]);

  const restartServerNow = useCallback(async () => {
    setRestarting(true);
    try {
      await api.restartServer(nodesetName);
      setStale(false);
      setSaveState("reloaded");
    } catch {
      /* surfaced via the api error wrapper */
    } finally {
      setRestarting(false);
    }
  }, [nodesetName]);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-1.5">
        <span className="text-xs font-semibold text-blue-300">{nodesetName}</span>
        {src && (
          <>
            <span className="text-[10px] text-gray-600">{src.file}</span>
            <span
              className={`rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${
                src.mode === "server"
                  ? "bg-amber-900/60 text-amber-300"
                  : "bg-gray-800 text-gray-500"
              }`}
            >
              {src.mode}
            </span>
          </>
        )}
        <div className="min-w-0 flex-1 truncate text-right text-[11px]">
          {saveError ? (
            <span className="text-red-400">{saveError}</span>
          ) : conflict ? (
            <span className="text-amber-400">
              File changed on disk.{" "}
              <button onClick={() => void load()} className="underline hover:text-amber-300">
                Reload
              </button>{" "}
              (discards edits)
            </span>
          ) : saveState === "reloaded" ? (
            <span className="text-green-400">Reloaded ✓</span>
          ) : saveState === "saved" ? (
            <span className="text-green-500">Saved — hot-reload pending…</span>
          ) : saveState === "deferred" ? (
            <span className="text-amber-400">Saved — reload deferred until the run ends</span>
          ) : dirty ? (
            <span className="text-gray-500">Unsaved changes (Ctrl+S)</span>
          ) : null}
        </div>
        <button
          onClick={() => void save()}
          disabled={!dirty || saveState === "saving"}
          className="shrink-0 rounded bg-blue-600 px-3 py-0.5 text-xs font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-800 disabled:text-gray-600"
        >
          {saveState === "saving" ? "Saving…" : "Save"}
        </button>
      </div>

      {/* Stale server banner */}
      {stale && (
        <div className="flex items-center gap-2 border-b border-amber-900/60 bg-amber-950/50 px-3 py-1 text-[11px] text-amber-300">
          <span className="flex-1">
            Server-mode nodeset — saved code is not live until the server restarts.
          </span>
          <button
            onClick={() => void restartServerNow()}
            disabled={restarting}
            className="flex shrink-0 items-center gap-1 rounded border border-amber-700 px-2 py-0.5 text-amber-200 hover:bg-amber-900/50 disabled:opacity-50"
          >
            <RefreshCw size={11} className={restarting ? "animate-spin" : ""} />
            {restarting ? "Restarting…" : "Restart server"}
          </button>
        </div>
      )}

      {/* Segments */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {loadError ? (
          <div className="px-3 py-3 text-xs text-red-400">{loadError}</div>
        ) : src === null ? (
          <div className="px-3 py-3 text-xs text-gray-600">Loading…</div>
        ) : (
          src.segments.map((seg, i) => (
            <div key={`${seg.kind}:${seg.name}:${seg.start_line}`}>
              <div className="flex items-center gap-2 border-y border-gray-800 bg-gray-950/60 px-3 py-1">
                <span
                  className={`rounded px-1.5 py-0 text-[9px] uppercase tracking-wider ${
                    KIND_STYLE[seg.kind] ?? "bg-gray-800 text-gray-400"
                  }`}
                >
                  {seg.kind}
                </span>
                <span className="text-[11px] text-gray-300">{seg.name}</span>
                <span className="text-[9px] text-gray-600">
                  L{seg.start_line}–{seg.end_line}
                </span>
              </div>
              <CodeMirrorEditor
                value={seg.text}
                docKey={`${src.file}:${seg.start_line}:${loadSeq}`}
                fill={false}
                onChange={(t) =>
                  setTexts((prev) => {
                    const next = [...prev];
                    next[i] = t;
                    return next;
                  })
                }
                onSave={() => void save()}
              />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
