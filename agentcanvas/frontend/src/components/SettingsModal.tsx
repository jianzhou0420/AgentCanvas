import { useEffect, useState } from "react";
import {
  X,
  Save,
  Eye,
  EyeOff,
  Check,
  ChevronDown,
  ChevronRight,
  Star,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import type { AppConfig, ProfilesState, ProviderDef } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

const TOP_PROVIDERS = ["openai", "anthropic", "google", "deepseek", "ollama"];

export default function SettingsModal({ open, onClose }: Props) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [profilesState, setProfilesState] = useState<ProfilesState | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  // Provider key inputs — only providers where user typed something
  const [keys, setKeys] = useState<Record<string, string>>({});
  // Track which providers already have a key saved on the server
  const [serverKeySet, setServerKeySet] = useState<Record<string, boolean>>({});

  // Active provider + model (now labeled "Default Model")
  const [activeProvider, setActiveProvider] = useState("");
  const [activeModel, setActiveModel] = useState("");

  // Per-provider model overrides (edit model name per configured provider)
  const [modelOverrides, setModelOverrides] = useState<Record<string, string>>(
    {},
  );

  // Ollama overrides
  const [ollamaBaseUrl, setOllamaBaseUrl] = useState("");

  // Custom provider overrides
  const [customKey, setCustomKey] = useState("");
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [customApiType, setCustomApiType] = useState("");

  // Provider model lists (fetched from backend)
  const [providerModels, setProviderModels] = useState<
    Record<string, string[]>
  >({});
  const [modelsLoading, setModelsLoading] = useState<Record<string, boolean>>(
    {},
  );

  // Collapsible more-providers section
  const [showMore, setShowMore] = useState(false);

  // Show/hide password per provider
  const [visibleKeys, setVisibleKeys] = useState<Record<string, boolean>>({});

  // App settings
  const [maxSteps, setMaxSteps] = useState(20);
  const [slamBackend, setSlamBackend] = useState("mock");
  const [ignoreLoopbackProxy, setIgnoreLoopbackProxy] = useState(true);

  const registry = profilesState?.registry ?? {};

  // Load config + profiles when modal opens
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError("");
    setSaved(false);
    setKeys({});
    setVisibleKeys({});
    setShowMore(false);
    setModelOverrides({});
    Promise.all([api.getConfig(), api.getProfiles()])
      .then(([cfg, ps]) => {
        setConfig(cfg);
        setProfilesState(ps);
        setMaxSteps(cfg.vlm_max_steps);
        setSlamBackend(cfg.slam_backend);
        setIgnoreLoopbackProxy(cfg.ignore_loopback_proxy ?? true);

        // Build server key set map
        const keySetMap: Record<string, boolean> = {};
        for (const [name, prof] of Object.entries(ps.profiles)) {
          keySetMap[name] = prof.api_key_set;
        }
        setServerKeySet(keySetMap);

        // Set active provider + model from current state
        const active = ps.active || "";
        setActiveProvider(active);
        if (active && ps.profiles[active]) {
          setActiveModel(ps.profiles[active].model);
        } else {
          setActiveModel("");
        }

        // Populate ollama/custom overrides from existing profiles
        const ollamaProf = ps.profiles["ollama"];
        if (ollamaProf) {
          setOllamaBaseUrl(ollamaProf.base_url);
        } else {
          setOllamaBaseUrl("");
        }
        const customProf = ps.profiles["custom"];
        if (customProf) {
          setCustomBaseUrl(customProf.base_url);
          setCustomModel(customProf.model);
          setCustomApiType(customProf.api_type);
          setCustomKey("");
        } else {
          setCustomBaseUrl("");
          setCustomModel("");
          setCustomApiType("");
          setCustomKey("");
        }
      })
      .catch(() => setError("Failed to load settings"))
      .finally(() => setLoading(false));
  }, [open]);

  // Fetch available models for saved providers
  useEffect(() => {
    if (!profilesState) return;
    const saved = Object.entries(profilesState.profiles)
      .filter(([, p]) => p.api_key_set || p.provider === "ollama")
      .map(([id]) => id);
    for (const id of saved) {
      setModelsLoading((prev) => ({ ...prev, [id]: true }));
      api
        .getProviderModels(id)
        .then((res) =>
          setProviderModels((prev) => ({ ...prev, [id]: res.models })),
        )
        .catch(() => {})
        .finally(() => setModelsLoading((prev) => ({ ...prev, [id]: false })));
    }
  }, [profilesState]);

  // Providers that can appear in the active dropdown:
  // providers with a key set on server OR with a dirty key typed, plus ollama (always)
  function getAvailableProviders(): string[] {
    const available: string[] = [];
    for (const id of Object.keys(registry)) {
      if (id === "ollama") {
        available.push(id);
        continue;
      }
      if (id === "custom") {
        if (serverKeySet[id] || customKey) available.push(id);
        continue;
      }
      if (serverKeySet[id] || keys[id]) available.push(id);
    }
    return available;
  }

  // Get list of configured models (providers with keys + ollama)
  function getConfiguredModels(): Array<{
    id: string;
    label: string;
    model: string;
    isDefault: boolean;
  }> {
    const models: Array<{
      id: string;
      label: string;
      model: string;
      isDefault: boolean;
    }> = [];
    for (const id of getAvailableProviders()) {
      const reg = registry[id];
      const prof = profilesState?.profiles[id];
      const label = reg?.label || id;
      const model =
        modelOverrides[id] ?? prof?.model ?? reg?.default_model ?? "";
      models.push({ id, label, model, isDefault: id === activeProvider });
    }
    return models;
  }

  function handleActiveProviderChange(providerId: string) {
    setActiveProvider(providerId);
    if (providerId) {
      // Auto-fill model from existing profile or registry default
      const overridden = modelOverrides[providerId];
      if (overridden !== undefined) {
        setActiveModel(overridden);
      } else {
        const existingProf = profilesState?.profiles[providerId];
        if (existingProf?.model) {
          setActiveModel(existingProf.model);
        } else {
          const reg = registry[providerId];
          setActiveModel(reg?.default_model || "");
        }
      }
    } else {
      setActiveModel("");
    }
  }

  function handleModelOverride(providerId: string, model: string) {
    setModelOverrides((prev) => ({ ...prev, [providerId]: model }));
    // If this is the active/default provider, also sync activeModel
    if (providerId === activeProvider) {
      setActiveModel(model);
    }
  }

  const handleSave = async () => {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      // Build overrides for ollama and custom
      const overrides: Record<string, Record<string, string>> = {};
      // Always send ollama override if it has a base_url or already exists
      if (ollamaBaseUrl || profilesState?.profiles["ollama"]) {
        overrides["ollama"] = { base_url: ollamaBaseUrl };
      }
      // Custom overrides
      if (
        customBaseUrl ||
        customModel ||
        customApiType ||
        customKey ||
        profilesState?.profiles["custom"]
      ) {
        overrides["custom"] = {
          base_url: customBaseUrl,
          model: customModel,
          api_type: customApiType,
        };
      }

      // Apply per-provider model overrides
      for (const [providerId, model] of Object.entries(modelOverrides)) {
        if (providerId === "custom" || providerId === "ollama") continue; // handled above
        if (!overrides[providerId]) overrides[providerId] = {};
        overrides[providerId].model = model;
      }
      // Also apply ollama model override if set
      if (modelOverrides["ollama"]) {
        if (!overrides["ollama"]) overrides["ollama"] = {};
        overrides["ollama"].model = modelOverrides["ollama"];
      }

      // Merge custom key into keys map
      const allKeys = { ...keys };
      if (customKey) {
        allKeys["custom"] = customKey;
      }

      await Promise.all([
        api.batchUpsertProfiles({
          keys: allKeys,
          active: activeProvider,
          active_model: activeModel,
          overrides,
        }),
        api.updateConfig({
          vlm_max_steps: maxSteps,
          slam_backend: slamBackend,
          ignore_loopback_proxy: ignoreLoopbackProxy,
        }),
      ]);

      // Re-fetch
      const [cfg, ps] = await Promise.all([api.getConfig(), api.getProfiles()]);
      setConfig(cfg);
      setProfilesState(ps);

      // Update server key set
      const keySetMap: Record<string, boolean> = {};
      for (const [name, prof] of Object.entries(ps.profiles)) {
        keySetMap[name] = prof.api_key_set;
      }
      setServerKeySet(keySetMap);
      setKeys({});
      setCustomKey("");
      setModelOverrides({});
      setActiveProvider(ps.active || "");
      if (ps.active && ps.profiles[ps.active]) {
        setActiveModel(ps.profiles[ps.active].model);
      }

      // Notify canvas that profiles changed so node dropdowns refresh
      window.dispatchEvent(new Event("profiles-changed"));

      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      setError("Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  const allProviderIds = Object.keys(registry);
  const moreProviders = allProviderIds.filter(
    (id) => !TOP_PROVIDERS.includes(id) && id !== "custom",
  );
  const configuredModels = getConfiguredModels();

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="mx-4 w-full max-w-lg rounded-lg border border-gray-700 bg-gray-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-800 px-5 py-3">
          <h2 className="text-lg font-semibold text-gray-100">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="max-h-[70vh] space-y-4 overflow-y-auto px-5 py-4">
          {loading && <p className="text-sm text-gray-400">Loading...</p>}
          {error && <p className="text-sm text-red-400">{error}</p>}

          {config && profilesState && (
            <>
              {/* ── Configured Models ── */}
              {configuredModels.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-medium text-gray-400">
                    Configured Models
                  </h3>
                  <p className="mb-2 text-xs text-gray-500">
                    Available in LLM/VLM node dropdowns. Click star to set
                    default.
                  </p>
                  <div className="space-y-1.5">
                    {configuredModels.map((m) => (
                      <div
                        key={m.id}
                        className="flex items-center gap-2 rounded border border-gray-700 bg-gray-800/40 px-3 py-1.5"
                      >
                        <button
                          onClick={() => handleActiveProviderChange(m.id)}
                          className={clsx(
                            "flex-shrink-0 transition-colors",
                            m.isDefault
                              ? "text-yellow-400"
                              : "text-gray-600 hover:text-gray-400",
                          )}
                          title={
                            m.isDefault ? "Default model" : "Set as default"
                          }
                        >
                          <Star
                            size={14}
                            fill={m.isDefault ? "currentColor" : "none"}
                          />
                        </button>
                        <span
                          className="w-28 flex-shrink-0 truncate text-sm font-medium text-gray-300"
                          title={m.label}
                        >
                          {m.label}
                        </span>
                        <ModelSelector
                          providerId={m.id}
                          model={m.model}
                          onChange={(v) => handleModelOverride(m.id, v)}
                          models={providerModels[m.id]}
                          loading={!!modelsLoading[m.id]}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {configuredModels.length === 0 && (
                <div className="rounded border border-gray-700 bg-gray-800/30 px-3 py-2">
                  <p className="text-sm text-gray-500">
                    No models configured. Add API keys below to get started.
                  </p>
                </div>
              )}

              <hr className="border-gray-700" />

              {/* ── Provider Keys ── */}
              <div>
                <h3 className="mb-3 text-sm font-medium text-gray-400">
                  API Keys
                </h3>
                <div className="space-y-2">
                  {/* Top providers (always visible) */}
                  {TOP_PROVIDERS.filter((id) => id !== "ollama").map((id) => (
                    <ProviderKeyRow
                      key={id}
                      providerId={id}
                      providerDef={registry[id]}
                      value={keys[id] ?? ""}
                      onChange={(v) =>
                        setKeys((prev) => ({ ...prev, [id]: v }))
                      }
                      hasServerKey={!!serverKeySet[id]}
                      visible={!!visibleKeys[id]}
                      onToggleVisible={() =>
                        setVisibleKeys((prev) => ({ ...prev, [id]: !prev[id] }))
                      }
                    />
                  ))}

                  {/* Ollama row — base URL instead of key */}
                  <div className="flex items-center gap-2">
                    <span
                      className="w-36 flex-shrink-0 truncate text-sm text-gray-300"
                      title="Ollama (local)"
                    >
                      Ollama (local)
                    </span>
                    <input
                      type="text"
                      value={ollamaBaseUrl}
                      onChange={(e) => setOllamaBaseUrl(e.target.value)}
                      placeholder="http://localhost:11434"
                      className="input-field flex-1 text-sm"
                    />
                    <span className="w-16 flex-shrink-0 text-right text-xs text-gray-500">
                      no key
                    </span>
                  </div>

                  {/* Collapsible more-providers */}
                  <button
                    onClick={() => setShowMore(!showMore)}
                    className="mt-1 flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300"
                  >
                    {showMore ? (
                      <ChevronDown size={14} />
                    ) : (
                      <ChevronRight size={14} />
                    )}
                    {showMore ? "Hide" : "More providers"} (
                    {moreProviders.length})
                  </button>

                  {showMore && (
                    <div className="ml-1 space-y-2 border-l-2 border-gray-800 pl-1">
                      {moreProviders.map((id) => (
                        <ProviderKeyRow
                          key={id}
                          providerId={id}
                          providerDef={registry[id]}
                          value={keys[id] ?? ""}
                          onChange={(v) =>
                            setKeys((prev) => ({ ...prev, [id]: v }))
                          }
                          hasServerKey={!!serverKeySet[id]}
                          visible={!!visibleKeys[id]}
                          onToggleVisible={() =>
                            setVisibleKeys((prev) => ({
                              ...prev,
                              [id]: !prev[id],
                            }))
                          }
                        />
                      ))}
                    </div>
                  )}

                  {/* Custom provider row */}
                  <div className="mt-2 space-y-2 rounded border border-gray-700 bg-gray-800/30 p-2">
                    <span className="text-sm font-medium text-gray-300">
                      Custom Provider
                    </span>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="password"
                        value={customKey}
                        onChange={(e) => setCustomKey(e.target.value)}
                        placeholder={
                          serverKeySet["custom"] ? "(key is set)" : "API key"
                        }
                        className="input-field text-sm"
                      />
                      <input
                        type="text"
                        value={customBaseUrl}
                        onChange={(e) => setCustomBaseUrl(e.target.value)}
                        placeholder="Base URL"
                        className="input-field text-sm"
                      />
                      <input
                        type="text"
                        value={customModel}
                        onChange={(e) => setCustomModel(e.target.value)}
                        placeholder="Model"
                        className="input-field text-sm"
                      />
                      <select
                        value={customApiType}
                        onChange={(e) => setCustomApiType(e.target.value)}
                        className="input-field text-sm"
                      >
                        <option value="">API type (auto)</option>
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="google">Google</option>
                        <option value="ollama">Ollama</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <hr className="border-gray-700" />

              {/* ── App Settings ── */}
              <div>
                <h3 className="mb-3 text-sm font-medium text-gray-400">
                  App Settings
                </h3>
                <div className="space-y-3">
                  <Field label="Max Steps">
                    <input
                      type="number"
                      value={maxSteps}
                      onChange={(e) =>
                        setMaxSteps(parseInt(e.target.value) || 1)
                      }
                      min={1}
                      max={100}
                      className="input-field w-24"
                    />
                  </Field>

                  <Field label="SLAM Backend">
                    <select
                      value={slamBackend}
                      onChange={(e) => setSlamBackend(e.target.value)}
                      className="input-field"
                    >
                      <option value="mock">Mock</option>
                      <option value="gaussian">Gaussian Splatting</option>
                    </select>
                  </Field>

                  <div>
                    <label className="flex items-start gap-2 text-sm font-medium text-gray-300">
                      <input
                        type="checkbox"
                        checked={ignoreLoopbackProxy}
                        onChange={(e) =>
                          setIgnoreLoopbackProxy(e.target.checked)
                        }
                        className="mt-0.5"
                      />
                      <span>
                        Ignore system HTTP proxy for in-process server calls
                        <span className="ml-1 block text-xs font-normal text-gray-500">
                          When on (default), backend↔auto-host child traffic
                          bypasses HTTP_PROXY / HTTPS_PROXY. Turn off only if
                          you intentionally route loopback traffic through a
                          proxy.
                        </span>
                      </span>
                    </label>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-gray-800 px-5 py-3">
          {saved && (
            <span className="flex items-center gap-1 text-sm text-green-400">
              <Check size={14} /> Saved
            </span>
          )}
          <button
            onClick={onClose}
            className="rounded border border-gray-700 px-4 py-1.5 text-sm text-gray-400 hover:border-gray-600 hover:text-gray-200"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || loading}
            className={clsx(
              "flex items-center gap-1.5 rounded px-4 py-1.5 text-sm font-medium",
              saving || loading
                ? "cursor-not-allowed bg-gray-700 text-gray-500"
                : "bg-blue-600 text-white hover:bg-blue-500",
            )}
          >
            <Save size={14} />
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ── */

function ProviderKeyRow({
  providerId,
  providerDef,
  value,
  onChange,
  hasServerKey,
  visible,
  onToggleVisible,
}: {
  providerId: string;
  providerDef: ProviderDef | undefined;
  value: string;
  onChange: (v: string) => void;
  hasServerKey: boolean;
  visible: boolean;
  onToggleVisible: () => void;
}) {
  const label = providerDef?.label || providerId;
  const dirty = value.length > 0;
  return (
    <div className="flex items-center gap-2">
      <span
        className="w-36 flex-shrink-0 truncate text-sm text-gray-300"
        title={label}
      >
        {label}
      </span>
      <div className="relative flex-1">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={hasServerKey ? "(key is set)" : "(paste API key)"}
          className="input-field w-full pr-8 text-sm"
        />
        <button
          type="button"
          onClick={onToggleVisible}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
        >
          {visible ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>
      <span
        className={clsx(
          "w-16 flex-shrink-0 text-right text-xs",
          dirty
            ? "text-yellow-400"
            : hasServerKey
              ? "text-green-400"
              : "text-gray-500",
        )}
      >
        {dirty ? "unsaved" : hasServerKey ? "saved" : "no key"}
      </span>
    </div>
  );
}

function ModelSelector({
  providerId,
  model,
  onChange,
  models,
  loading,
}: {
  providerId: string;
  model: string;
  onChange: (v: string) => void;
  models: string[] | undefined;
  loading: boolean;
}) {
  if (models && models.length > 0) {
    return (
      <select
        value={models.includes(model) ? model : "__current__"}
        onChange={(e) =>
          onChange(e.target.value === "__current__" ? model : e.target.value)
        }
        className="input-field flex-1 text-sm"
      >
        {!models.includes(model) && model && (
          <option value="__current__">{model}</option>
        )}
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    );
  }

  return (
    <input
      type="text"
      value={model}
      onChange={(e) => onChange(e.target.value)}
      placeholder={loading ? "Loading models..." : "model name"}
      className="input-field flex-1 text-sm"
      disabled={loading}
    />
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        {label}
      </label>
      {children}
    </div>
  );
}
