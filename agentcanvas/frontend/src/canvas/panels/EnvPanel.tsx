/** Generic env panel panel — renders any nodeset's BaseEnvPanel.
 *
 * Replaces the env-specific EnvContextBar. The panel queries the backend
 * for the list of registered env panels, lets the user pick one, then
 * renders that env panel's declared `fields` and `actions`. There is no
 * env-specific code here — adding a new env nodeset means writing a
 * BaseEnvPanel subclass inside that nodeset's file. Done.
 *
 * Action buttons send their `side_effect` (run_start/run_pause/run_stop)
 * to the existing run-lifecycle code in the store. The env panel never
 * reaches into LoopRunner directly.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Globe,
  Loader2,
  Play,
  Pause,
  Square,
  RotateCcw,
  SkipForward,
  type LucideIcon,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../../api";
import { useStore } from "../../store";
import { useFlowStore } from "../useFlowStore";
import { runPipeline } from "../runPipeline";
import type { EnvPanelInfo, EnvPanelOption, EnvPanelState } from "../../types";

const ACTIVE_KEY = "agentcanvas:active_env_panel";

const ACTION_ICONS: Record<string, LucideIcon> = {
  play: Play,
  pause: Pause,
  stop: Square,
  step: SkipForward,
  reset: RotateCcw,
};

const ACTION_COLORS: Record<string, string> = {
  play: "bg-green-600 hover:bg-green-500 text-white",
  pause: "bg-yellow-600 hover:bg-yellow-500 text-white",
  stop: "bg-red-600 hover:bg-red-500 text-white",
  reset: "bg-orange-600 hover:bg-orange-500 text-white",
  step: "bg-blue-600 hover:bg-blue-500 text-white",
};

export default function EnvPanel() {
  const navStatus = useStore((s) => s.navStatus);
  const navPause = useStore((s) => s.navPause);
  const navStop = useStore((s) => s.navStop);

  const [available, setAvailable] = useState<EnvPanelInfo[]>([]);
  const [activeName, setActiveName] = useState<string>(
    () => localStorage.getItem(ACTIVE_KEY) || "",
  );
  const [state, setState] = useState<EnvPanelState | null>(null);
  const [optionsByField, setOptionsByField] = useState<
    Record<string, EnvPanelOption[]>
  >({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const active = useMemo(
    () => available.find((c) => c.name === activeName) || null,
    [available, activeName],
  );

  const isExecuting = navStatus === "running" || navStatus === "loading";
  const isAvailable = state ? state["available"] !== false : false;

  // Refresh the env panel list (called on mount and on nodeset reload)
  const refreshList = useCallback(async () => {
    try {
      const list = await api.envPanelList();
      setAvailable(list);
      // Auto-select the first available env panel if nothing is chosen yet,
      // or if the previously selected one no longer exists.
      if (list.length > 0 && !list.find((c) => c.name === activeName)) {
        setActiveName(list[0].name);
      } else if (list.length === 0) {
        setActiveName("");
      }
    } catch {
      setAvailable([]);
    }
  }, [activeName]);

  useEffect(() => {
    refreshList();
    const handler = () => refreshList();
    window.addEventListener("nodesets-changed", handler);
    return () => window.removeEventListener("nodesets-changed", handler);
  }, [refreshList]);

  // Load state and options whenever the active env panel changes
  useEffect(() => {
    if (!activeName) {
      setState(null);
      setOptionsByField({});
      localStorage.removeItem(ACTIVE_KEY);
      return;
    }
    localStorage.setItem(ACTIVE_KEY, activeName);
    let cancelled = false;
    (async () => {
      try {
        const s = await api.envPanelState(activeName);
        if (!cancelled) setState(s);
        const ctrl = available.find((c) => c.name === activeName);
        if (ctrl) {
          const opts: Record<string, EnvPanelOption[]> = {};
          for (const f of ctrl.fields) {
            if (f.kind === "select") {
              try {
                opts[f.name] = await api.envPanelOptions(activeName, f.name);
              } catch {
                opts[f.name] = [];
              }
            }
          }
          if (!cancelled) setOptionsByField(opts);
        }
      } catch (e) {
        if (!cancelled) {
          setState(null);
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeName, available]);

  const handleFieldChange = async (field: string, value: unknown) => {
    if (!activeName || isExecuting) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.envPanelSetField(activeName, field, value);
      setState(next);
      // Some field changes (e.g. split) can change which options are valid
      // for other fields — refresh dynamic options.
      const ctrl = available.find((c) => c.name === activeName);
      if (ctrl) {
        const opts: Record<string, EnvPanelOption[]> = { ...optionsByField };
        for (const f of ctrl.fields) {
          if (f.kind === "select" && f.name !== field) {
            try {
              opts[f.name] = await api.envPanelOptions(activeName, f.name);
            } catch {
              /* ignore */
            }
          }
        }
        setOptionsByField(opts);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Field change failed");
    } finally {
      setBusy(false);
    }
  };

  const handleAction = async (action: string, sideEffect: string) => {
    if (!activeName) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.envPanelAction(activeName, action, {});
      if (!result.ok) {
        setError(result.error || `Action ${action} failed`);
        return;
      }
      // Interpret side_effect: dispatch the matching run-lifecycle call.
      switch (sideEffect) {
        case "run_start": {
          const graph = useFlowStore.getState().getGraphForExecution();
          if (!graph) return;
          const executionId = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
          useFlowStore.getState().startExecution(executionId);
          await runPipeline(graph, executionId);
          break;
        }
        case "run_pause":
          await navPause();
          break;
        case "run_stop":
          await navStop();
          break;
        default:
          // no follow-up; just refresh state
          break;
      }
      // Refresh state after the action completes (to pick up env mutations).
      try {
        const next = await api.envPanelState(activeName);
        setState(next);
      } catch {
        /* ignore */
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : `Action ${action} failed`);
    } finally {
      setBusy(false);
    }
  };

  // Empty state: no env panels loaded
  if (available.length === 0) {
    return (
      <div className="flex items-center gap-2 border-b border-gray-800/50 bg-gray-900/60 px-3 py-1">
        <Globe size={12} className="text-gray-600" />
        <span className="text-[10px] text-gray-600">
          No env panel — load an environment nodeset to enable a control panel
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 border-b border-gray-800/50 bg-gray-900/60 px-3 py-1">
      <Globe size={12} className="text-emerald-500" />

      {/* Picker dropdown — multiple env panels possible */}
      <select
        value={activeName}
        onChange={(e) => setActiveName(e.target.value)}
        disabled={isExecuting}
        className={clsx(
          "rounded bg-gray-800 px-1.5 py-0.5 text-[10px] font-medium outline-none",
          isExecuting
            ? "text-gray-600"
            : "text-emerald-300 hover:bg-gray-700 focus:ring-1 focus:ring-blue-500",
        )}
        title="Active env panel"
      >
        {available.map((c) => (
          <option key={c.name} value={c.name}>
            {c.display_name}
          </option>
        ))}
      </select>

      {!isAvailable && state && (
        <>
          <AlertCircle size={10} className="text-yellow-600" />
          <span className="text-[10px] text-yellow-600/80">
            {(state["message"] as string) || "Env panel unavailable"}
          </span>
        </>
      )}

      {isAvailable && active && (
        <>
          <div className="mx-1 h-3 border-l border-gray-700" />
          {active.fields.map((f) => (
            <FieldRenderer
              key={f.name}
              field={f}
              value={state?.[f.name]}
              options={optionsByField[f.name] || []}
              disabled={isExecuting || busy}
              onChange={(v) => handleFieldChange(f.name, v)}
            />
          ))}

          <div className="mx-1 h-3 border-l border-gray-700" />

          {/* Action buttons */}
          {active.actions.map((a) => {
            const Icon = ACTION_ICONS[a.name];
            const color =
              ACTION_COLORS[a.name] ||
              "bg-gray-700 hover:bg-gray-600 text-gray-200";
            const isPlay = a.side_effect === "run_start";
            const isPauseStop =
              a.side_effect === "run_pause" || a.side_effect === "run_stop";
            const buttonDisabled =
              busy || (isPlay && isExecuting) || (isPauseStop && !isExecuting);
            return (
              <button
                key={a.name}
                onClick={() => handleAction(a.name, a.side_effect)}
                disabled={buttonDisabled}
                className={clsx(
                  "flex items-center justify-center rounded px-2 py-1 text-[10px]",
                  buttonDisabled ? "bg-gray-800 text-gray-600" : color,
                )}
                title={a.label}
              >
                {Icon ? <Icon size={11} /> : a.label}
              </button>
            );
          })}

          {/* Optional: episode info preview */}
          {state && state["current_episode"] ? (
            <CurrentEpisodePreview
              info={state["current_episode"] as Record<string, unknown>}
            />
          ) : null}
        </>
      )}

      {error && (
        <>
          <div className="mx-1 h-3 border-l border-gray-700" />
          <AlertCircle size={10} className="text-red-400" />
          <span className="text-[10px] text-red-400">{error}</span>
        </>
      )}

      {busy && <Loader2 size={11} className="animate-spin text-blue-400" />}
    </div>
  );
}

// ── Field renderers ──

function FieldRenderer({
  field,
  value,
  options,
  disabled,
  onChange,
}: {
  field: import("../../types").EnvPanelField;
  value: unknown;
  options: EnvPanelOption[];
  disabled: boolean;
  onChange: (value: unknown) => void;
}) {
  if (field.kind === "select") {
    const stringValue = value == null ? "" : String(value);
    return (
      <>
        <span className="text-[10px] text-gray-500">{field.label}</span>
        <select
          value={stringValue}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={clsx(
            "rounded bg-gray-800 px-1.5 py-0.5 text-[10px] outline-none",
            disabled
              ? "text-gray-600"
              : "text-gray-300 hover:bg-gray-700 focus:ring-1 focus:ring-blue-500",
          )}
        >
          {options.length === 0 && stringValue && (
            <option value={stringValue}>{stringValue}</option>
          )}
          {options.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>
              {o.label}
            </option>
          ))}
        </select>
      </>
    );
  }
  if (field.kind === "number") {
    const numValue = typeof value === "number" ? value : Number(value ?? 0);
    return (
      <>
        <span className="text-[10px] text-gray-500">{field.label}</span>
        <input
          type="number"
          value={Number.isFinite(numValue) ? numValue : 0}
          min={field.min ?? undefined}
          max={field.max ?? undefined}
          step={field.step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={clsx(
            "w-16 rounded bg-gray-800 px-1.5 py-0.5 text-center font-mono text-[10px] outline-none",
            disabled
              ? "text-gray-600"
              : "text-gray-300 focus:ring-1 focus:ring-blue-500",
          )}
        />
      </>
    );
  }
  // text / slider — minimal fallback
  return (
    <>
      <span className="text-[10px] text-gray-500">{field.label}</span>
      <input
        type={field.kind === "slider" ? "range" : "text"}
        value={value == null ? "" : String(value)}
        min={field.min ?? undefined}
        max={field.max ?? undefined}
        step={field.step ?? undefined}
        onChange={(e) =>
          onChange(
            field.kind === "slider" ? Number(e.target.value) : e.target.value,
          )
        }
        disabled={disabled}
        placeholder={field.placeholder ?? ""}
        className="rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-300 outline-none focus:ring-1 focus:ring-blue-500"
      />
    </>
  );
}

function CurrentEpisodePreview({ info }: { info: Record<string, unknown> }) {
  const scene = info["scene_id"] as string | undefined;
  const instruction = info["instruction"] as string | undefined;
  if (!scene && !instruction) return null;
  const scenePart = scene ? scene.split("/").pop()?.replace(".glb", "") : "";
  const instrPart =
    instruction && instruction.length > 80
      ? instruction.slice(0, 80) + "…"
      : instruction;
  return (
    <div className="ml-2 flex items-center gap-2 overflow-hidden">
      {scenePart && (
        <span className="whitespace-nowrap text-[10px] text-gray-600">
          {scenePart}
        </span>
      )}
      {instrPart && (
        <span
          className="truncate text-[10px] text-gray-500 italic"
          title={instruction}
        >
          {instrPart}
        </span>
      )}
    </div>
  );
}
