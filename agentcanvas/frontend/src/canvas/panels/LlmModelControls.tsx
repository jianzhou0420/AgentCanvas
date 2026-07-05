/** LLM model controls for the properties panel.
 *
 * ModelRefPicker — one dropdown, three reference modes: Default (follow the
 * active profile), a named profile, or Browse… (provider → model cascade
 * pinned inline on the node as {provider, model}).
 *
 * useModelCapabilities + CapAnnotatedField — edit-time rendering of the
 * parameter rulebook: the panel fetches GET /api/providers/{id}/capabilities
 * for the resolved (provider, model) and imposes the verdicts on the
 * controls (locked shows the value + reason, range clamps the slider,
 * unsupported grays out, min_hint warns). Same rulebook finalize applies at
 * call time — the panel is just the "before" render of it.
 */

import { useEffect, useState } from "react";
import { api } from "../../api";
import type { Capabilities, ProvidersMap } from "../../types";
import { ConfigFieldRenderer } from "../nodes/agentloop/inner/layouts/ConfigFieldRenderer";
import type { ConfigFieldSchema } from "../nodes/agentloop/inner/layouts/layoutUtils";
import { useFlowStore } from "../useFlowStore";

/* ── Resolve node → (provider, model) and fetch rulebook verdicts ── */

export function useModelCapabilities(
  data: Record<string, unknown>,
  enabled: boolean,
): Capabilities | null {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const profile = String(data.profile ?? "");
  const provider = String(data.provider ?? "");
  const model = String(data.model ?? "");

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    (async () => {
      try {
        let resolved: { provider: string; model: string } | null = null;
        if (profile === "__direct__") {
          if (provider && model) resolved = { provider, model };
        } else {
          const ps = await api.getProfiles();
          const name = profile || ps.active;
          const p = name ? ps.profiles[name] : undefined;
          if (p) resolved = { provider: p.provider, model: p.model };
        }
        if (!resolved) {
          if (!cancelled) setCaps(null);
          return;
        }
        const res = await api.getProviderCapabilities(
          resolved.provider,
          resolved.model,
        );
        if (!cancelled) setCaps(res.capabilities);
      } catch {
        if (!cancelled) setCaps(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled, profile, provider, model]);

  return caps;
}

/* ── Rulebook-annotated config field ── */

export function CapAnnotatedField({
  field,
  data,
  nodeId,
  caps,
}: {
  field: ConfigFieldSchema;
  data: Record<string, unknown>;
  nodeId: string;
  caps: Capabilities | null;
}) {
  const verdicts = caps?.[field.name] ?? [];
  const locked = verdicts.find((v) => v.kind === "locked");
  const range = verdicts.find((v) => v.kind === "range");
  const unsupported = verdicts.find((v) => v.kind === "unsupported");
  const advisory = verdicts.find(
    (v) => v.kind === "min_hint" || v.kind === "required",
  );

  // Locked is never hidden: show the value and the reason instead of a widget.
  if (locked) {
    return (
      <div className="text-[9px] text-gray-400">
        <span className="flex items-center justify-between">
          <span>
            {field.label}: <span className="text-gray-200">{String(locked.value)}</span> 🔒
          </span>
        </span>
        <div className="text-[8px] italic text-gray-500">{locked.note}</div>
      </div>
    );
  }

  let effective = field;
  if (range && Array.isArray(range.value) && range.value.length === 2) {
    effective = {
      ...field,
      min: Number(range.value[0]),
      max: Number(range.value[1]),
    };
  }

  return (
    <div className={unsupported ? "opacity-50" : undefined}>
      <ConfigFieldRenderer field={effective} data={data} nodeId={nodeId} />
      {unsupported && (
        <div className="text-[8px] italic text-gray-500">
          not supported here — {unsupported.note}
        </div>
      )}
      {!unsupported && advisory && (
        <div className="text-[8px] italic text-amber-500/80">
          {advisory.kind === "required"
            ? `unset → ${String(advisory.value)} will be injected (${advisory.note})`
            : `hint: ≥ ${String(advisory.value)} — ${advisory.note}`}
        </div>
      )}
      {!unsupported && range && (
        <div className="text-[8px] italic text-gray-500">{range.note}</div>
      )}
    </div>
  );
}

/* ── Three-mode model picker ── */

export function ModelRefPicker({
  field,
  data,
  nodeId,
}: {
  field: ConfigFieldSchema;
  data: Record<string, unknown>;
  nodeId: string;
}) {
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const current = String(data.profile ?? "");
  const [browsing, setBrowsing] = useState(current === "__direct__");
  const [providers, setProviders] = useState<ProvidersMap | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [pickProvider, setPickProvider] = useState(
    String(data.provider ?? "openai"),
  );
  const [pickModel, setPickModel] = useState(String(data.model ?? ""));

  // Lazy-load the provider registry when Browse opens
  useEffect(() => {
    if (!browsing || providers) return;
    api
      .getProviders()
      .then(setProviders)
      .catch(() => {});
  }, [browsing, providers]);

  // Live model list for the picked provider (when its key is usable)
  useEffect(() => {
    if (!browsing || !providers) return;
    setModels([]);
    const info = providers[pickProvider];
    if (!info || (!info.key_set && pickProvider !== "ollama")) return;
    api
      .getProviderModels(pickProvider)
      .then((res) => setModels(res.models))
      .catch(() => {});
  }, [browsing, providers, pickProvider]);

  const options = field.options ?? [];

  const applyDirect = () => {
    const model = pickModel || providers?.[pickProvider]?.default_model || "";
    if (!model) return;
    updateNodeData(nodeId, {
      profile: "__direct__",
      provider: pickProvider,
      model,
    });
    setBrowsing(false);
  };

  return (
    <div className="nopan nodrag space-y-1 text-[9px] text-gray-400">
      <label className="block">
        {field.label}
        <select
          value={browsing || current === "__direct__" ? "__browse__" : current}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "__browse__") {
              setBrowsing(true);
              return;
            }
            setBrowsing(false);
            updateNodeData(nodeId, { profile: v });
          }}
          className="mt-0.5 w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.value === "" ? `↪ ${opt.label}` : opt.label}
            </option>
          ))}
          <option value="__browse__">
            {current === "__direct__" && data.model
              ? `pinned: ${data.model} · ${data.provider}`
              : "Browse models…"}
          </option>
        </select>
      </label>

      {browsing && (
        <div className="space-y-1 rounded border border-gray-700 bg-gray-800/50 p-1.5">
          <select
            value={pickProvider}
            onChange={(e) => setPickProvider(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
          >
            {providers ? (
              Object.entries(providers).map(([id, info]) => (
                <option key={id} value={id}>
                  {info.label}
                  {!info.key_set && id !== "ollama"
                    ? ` (no key — set ${info.key_env})`
                    : ""}
                </option>
              ))
            ) : (
              <option value={pickProvider}>loading providers…</option>
            )}
          </select>
          {models.length > 0 ? (
            <select
              value={pickModel || providers?.[pickProvider]?.default_model || ""}
              onChange={(e) => setPickModel(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={pickModel}
              onChange={(e) => setPickModel(e.target.value)}
              placeholder={`model (default: ${
                providers?.[pickProvider]?.default_model || "—"
              })`}
              className="w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
            />
          )}
          <div className="flex justify-end gap-1.5">
            <button
              type="button"
              onClick={() => setBrowsing(false)}
              className="rounded border border-gray-700 px-1.5 py-0.5 text-[9px] text-gray-500 hover:text-gray-300"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={applyDirect}
              className="rounded bg-blue-600 px-1.5 py-0.5 text-[9px] text-white hover:bg-blue-500"
            >
              Pin on node
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
