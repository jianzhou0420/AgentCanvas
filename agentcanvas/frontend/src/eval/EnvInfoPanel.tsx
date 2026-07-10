import { useEffect, useRef, useState, useCallback } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import type {
  EnvPanelInfo,
  EnvPanelOption,
  EnvPanelField as EnvPanelFieldSchema,
} from "../types";
import type { GraphIntrospection } from "./types";

interface Props {
  introspection: GraphIntrospection | null;
  selectors: Record<string, string | number>;
  episodeCount: number;
  startEpisodeIndex: number;
  disabled: boolean;
  onSelectorsChange: (selectors: Record<string, string | number>) => void;
  onEpisodeCountChange: (n: number) => void;
  onStartEpisodeChange: (n: number) => void;
  onLoadNodeset?: () => Promise<void>;
}

const inputCls =
  "mt-1 w-full bg-gray-800 text-gray-200 text-sm px-2 py-1 rounded border border-gray-700";

// `episode_index` is iterated by the eval batch runner itself — exclude it
// from the cascade panel so users don't pick a single episode here.
const EPISODE_FIELD = "episode_index";

export default function EnvInfoPanel({
  introspection,
  selectors,
  episodeCount,
  startEpisodeIndex,
  disabled,
  onSelectorsChange,
  onEpisodeCountChange,
  onStartEpisodeChange,
  onLoadNodeset,
}: Props) {
  const [panel, setPanel] = useState<EnvPanelInfo | null>(null);
  const [optionsByField, setOptionsByField] = useState<
    Record<string, EnvPanelOption[]>
  >({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const envNodeset = introspection?.env_nodeset || null;
  const loaded = introspection?.loaded ?? false;
  const metadata = introspection?.metadata ?? null;

  const prevEnvRef = useRef<string | null>(null);

  // Load panel schema + initial state whenever the env nodeset becomes
  // available (graph switched or user clicked "Load Nodeset").
  useEffect(() => {
    if (!envNodeset || !loaded) {
      setPanel(null);
      setOptionsByField({});
      return;
    }
    const envChanged = prevEnvRef.current !== envNodeset;
    prevEnvRef.current = envNodeset;

    let cancelled = false;
    (async () => {
      setBusy(true);
      try {
        const list = await api.envPanelList();
        const ctrl = list.find((c) => c.name === envNodeset) || null;
        if (!ctrl) {
          if (!cancelled) {
            setPanel(null);
            setOptionsByField({});
            setError(`No panel registered for '${envNodeset}'`);
          }
          return;
        }
        const state = await api.envPanelState(envNodeset);
        const cascadeFields = ctrl.fields.filter(
          (f) => f.name !== EPISODE_FIELD,
        );

        const seeded: Record<string, string | number> = {};
        for (const f of cascadeFields) {
          const v = state[f.name];
          if (v === undefined || v === null || v === "") continue;
          seeded[f.name] = typeof v === "number" ? v : String(v);
        }

        const opts: Record<string, EnvPanelOption[]> = {};
        for (const f of cascadeFields) {
          if (f.kind !== "select") continue;
          try {
            opts[f.name] = await api.envPanelOptions(envNodeset, f.name);
          } catch {
            opts[f.name] = [];
          }
        }

        if (cancelled) return;
        setPanel(ctrl);
        setOptionsByField(opts);
        setError(null);
        // Re-seed parent selectors only when the env nodeset itself changed
        // (graph swap). Otherwise preserve user's in-progress selection.
        if (envChanged) onSelectorsChange(seeded);
      } catch (e) {
        if (!cancelled) {
          setPanel(null);
          setOptionsByField({});
          setError(e instanceof Error ? e.message : "Failed to load panel");
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // onSelectorsChange is intentionally excluded — it's a parent setter and
    // including it would cause re-fetch loops on every parent re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [envNodeset, loaded]);

  const refreshDownstreamOptions = useCallback(
    async (changedField: string, ctrl: EnvPanelInfo) => {
      if (!envNodeset) return;
      const opts: Record<string, EnvPanelOption[]> = { ...optionsByField };
      for (const f of ctrl.fields) {
        if (f.kind !== "select" || f.name === changedField) continue;
        if (f.name === EPISODE_FIELD) continue;
        try {
          opts[f.name] = await api.envPanelOptions(envNodeset, f.name);
        } catch {
          /* ignore — keep stale options rather than blanking */
        }
      }
      setOptionsByField(opts);
    },
    [envNodeset, optionsByField],
  );

  const handleFieldChange = useCallback(
    async (field: string, value: string | number) => {
      if (!envNodeset || disabled || !panel) return;
      setBusy(true);
      try {
        // Push to backend so downstream options (e.g. SIMPLER's `task_id`
        // depends on `split`) refresh against the new state.
        const next = await api.envPanelSetField(envNodeset, field, value);
        // The backend may have normalized the value (e.g. coerced an
        // unknown split back to the default) and reset downstream fields
        // (e.g. clearing task_id when split changes). Re-derive selectors
        // from the returned state so the panel matches what the panel
        // will actually use during eval.
        const cascadeFields = panel.fields.filter(
          (f) => f.name !== EPISODE_FIELD,
        );
        const updated: Record<string, string | number> = {};
        for (const f of cascadeFields) {
          const v = next[f.name];
          if (v === undefined || v === null || v === "") continue;
          updated[f.name] = typeof v === "number" ? v : String(v);
        }
        onSelectorsChange(updated);
        await refreshDownstreamOptions(field, panel);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Field change failed");
      } finally {
        setBusy(false);
      }
    },
    [envNodeset, disabled, panel, onSelectorsChange, refreshDownstreamOptions],
  );

  if (!introspection) return null;

  if (!envNodeset) {
    return (
      <div className="flex items-center gap-1.5 rounded border border-yellow-800 bg-yellow-900/20 px-2 py-1.5 text-xs text-yellow-400">
        <AlertTriangle size={12} />
        No environment nodeset detected in this graph
      </div>
    );
  }

  if (!loaded) {
    return (
      <LoadNodesetPrompt envNodeset={envNodeset} onLoadNodeset={onLoadNodeset} />
    );
  }

  const cascadeFields = panel
    ? panel.fields.filter((f) => f.name !== EPISODE_FIELD)
    : [];

  // Best-effort total-episodes hint: prefer episode_counts[split] from the
  // legacy metadata; fall back to nothing.
  const splitValue = selectors["split"];
  const totalForSplit =
    typeof splitValue === "string"
      ? metadata?.episode_counts?.[splitValue]
      : undefined;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-xs text-gray-400">
        <span>Env:</span>
        <span className="font-mono text-gray-300">
          {metadata?.env_name ?? envNodeset}
        </span>
        {busy && <Loader2 size={11} className="animate-spin text-blue-400" />}
      </div>

      {cascadeFields.map((f) => (
        <CascadeFieldRenderer
          key={f.name}
          field={f}
          value={selectors[f.name]}
          options={optionsByField[f.name] || []}
          disabled={disabled || busy}
          onChange={(v) => handleFieldChange(f.name, v)}
        />
      ))}

      <label className="text-xs text-gray-400">
        Episodes
        <input
          type="number"
          min={-1}
          value={episodeCount}
          onChange={(e) => onEpisodeCountChange(Number(e.target.value))}
          disabled={disabled}
          className={inputCls}
        />
        <span className={clsx("text-xs", "text-gray-500")}>
          -1 = all
          {totalForSplit !== undefined ? ` (${totalForSplit} available)` : ""}
        </span>
      </label>

      <label className="text-xs text-gray-400">
        Start from episode
        <input
          type="number"
          min={0}
          value={startEpisodeIndex}
          onChange={(e) =>
            onStartEpisodeChange(Math.max(0, Number(e.target.value) || 0))
          }
          disabled={disabled}
          className={inputCls}
        />
      </label>

      {metadata?.metrics && metadata.metrics.length > 0 && (
        <div className="text-xs text-gray-500">
          Metrics: {metadata.metrics.join(", ")}
        </div>
      )}

      {error && (
        <div className="rounded border border-red-800 bg-red-900/20 px-2 py-1 text-xs text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}

// ── "Load Nodeset" prompt — server-mode env loads spawn a subprocess and can
//     take tens of seconds, so show an in-flight state and never fail silently. ──

function LoadNodesetPrompt({
  envNodeset,
  onLoadNodeset,
}: {
  envNodeset: string;
  onLoadNodeset?: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClick = async () => {
    if (!onLoadNodeset || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onLoadNodeset();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded border border-orange-800 bg-orange-900/20 p-2">
      <div className="flex items-center gap-1.5 text-xs text-orange-400">
        <AlertTriangle size={12} />
        Nodeset <span className="font-mono">{envNodeset}</span> not loaded
      </div>
      {onLoadNodeset && (
        <button
          onClick={handleClick}
          disabled={busy}
          className="flex items-center justify-center gap-1.5 rounded bg-orange-700 px-2 py-1 text-xs text-white hover:bg-orange-600 disabled:cursor-not-allowed disabled:opacity-70"
        >
          {busy && <Loader2 size={12} className="animate-spin" />}
          {busy ? "Loading… (server may take ~30s)" : "Load Nodeset"}
        </button>
      )}
      {error && (
        <div className="rounded border border-red-800 bg-red-900/20 px-2 py-1 text-xs break-words text-red-400">
          Load failed — {error}
        </div>
      )}
    </div>
  );
}

// ── Field renderers (eval-page sized — full-width, vertical stack) ──

function CascadeFieldRenderer({
  field,
  value,
  options,
  disabled,
  onChange,
}: {
  field: EnvPanelFieldSchema;
  value: string | number | undefined;
  options: EnvPanelOption[];
  disabled: boolean;
  onChange: (v: string | number) => void;
}) {
  if (field.kind === "select") {
    const stringValue = value == null ? "" : String(value);
    return (
      <label className="text-xs text-gray-400">
        {field.label}
        <select
          value={stringValue}
          onChange={(e) => {
            const opt = options.find((o) => String(o.value) === e.target.value);
            onChange(opt ? opt.value : e.target.value);
          }}
          disabled={disabled}
          className={inputCls}
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
      </label>
    );
  }

  if (field.kind === "number") {
    const n = typeof value === "number" ? value : Number(value ?? 0);
    return (
      <label className="text-xs text-gray-400">
        {field.label}
        <input
          type="number"
          value={Number.isFinite(n) ? n : 0}
          min={field.min ?? undefined}
          max={field.max ?? undefined}
          step={field.step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={inputCls}
        />
      </label>
    );
  }

  // text / slider — minimal fallback
  return (
    <label className="text-xs text-gray-400">
      {field.label}
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
        className={inputCls}
      />
    </label>
  );
}
