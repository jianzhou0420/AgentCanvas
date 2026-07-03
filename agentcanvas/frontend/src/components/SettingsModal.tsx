/** Settings modal — two sections mirroring the two user-facing planes.
 *
 * Credentials: one row per provider; keys persist to ~/.agentcanvas/.keys
 * via PUT /api/providers/{id}/key (badge shows file / env / none), with a
 * Validate round trip. Model profiles: the store's actual model — named
 * {provider, model} rows with CRUD + ★ active. Every action applies
 * immediately (plain CRUD; the legacy batch endpoint is gone).
 */

import { useCallback, useEffect, useState } from "react";
import {
  X,
  Save,
  Eye,
  EyeOff,
  Check,
  ChevronDown,
  ChevronRight,
  Star,
  Trash2,
  Plus,
  ShieldCheck,
  Loader2,
} from "lucide-react";
import clsx from "clsx";
import { api } from "../api";
import type { AppConfig, LLMProfile, ProfilesState, ProvidersMap } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

const TOP_PROVIDERS = ["openai", "anthropic", "google", "deepseek", "ollama"];

export default function SettingsModal({ open, onClose }: Props) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [providers, setProviders] = useState<ProvidersMap | null>(null);
  const [profilesState, setProfilesState] = useState<ProfilesState | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [showMore, setShowMore] = useState(false);

  // App settings
  const [maxSteps, setMaxSteps] = useState(20);
  const [slamBackend, setSlamBackend] = useState("mock");
  const [ignoreLoopbackProxy, setIgnoreLoopbackProxy] = useState(true);

  const reload = useCallback(async () => {
    const [cfg, ps, prov] = await Promise.all([
      api.getConfig(),
      api.getProfiles(),
      api.getProviders(),
    ]);
    setConfig(cfg);
    setProfilesState(ps);
    setProviders(prov);
    setMaxSteps(cfg.vlm_max_steps);
    setSlamBackend(cfg.slam_backend);
    setIgnoreLoopbackProxy(cfg.ignore_loopback_proxy ?? true);
  }, []);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError("");
    setSaved(false);
    setShowMore(false);
    reload()
      .catch(() => setError("Failed to load settings"))
      .finally(() => setLoading(false));
  }, [open, reload]);

  const notifyProfilesChanged = () => {
    window.dispatchEvent(new Event("profiles-changed"));
  };

  const handleSaveAppSettings = async () => {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      await api.updateConfig({
        vlm_max_steps: maxSteps,
        slam_backend: slamBackend,
        ignore_loopback_proxy: ignoreLoopbackProxy,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      setError("Failed to save settings");
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  const providerIds = providers ? Object.keys(providers) : [];
  const moreProviders = providerIds.filter((id) => !TOP_PROVIDERS.includes(id));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="mx-4 w-full max-w-2xl rounded-lg border border-gray-700 bg-gray-900 shadow-2xl"
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
        <div className="max-h-[72vh] space-y-5 overflow-y-auto px-5 py-4">
          {loading && <p className="text-sm text-gray-400">Loading...</p>}
          {error && <p className="text-sm text-red-400">{error}</p>}

          {config && providers && profilesState && (
            <>
              {/* ── Model Profiles ── */}
              <ProfilesSection
                profilesState={profilesState}
                providers={providers}
                onChanged={async () => {
                  await reload();
                  notifyProfilesChanged();
                }}
                onError={setError}
              />

              <hr className="border-gray-700" />

              {/* ── Credentials ── */}
              <div>
                <h3 className="mb-1 text-sm font-medium text-gray-400">
                  Credentials
                </h3>
                <p className="mb-2 text-xs text-gray-500">
                  Keys are stored in{" "}
                  <code className="text-gray-400">~/.agentcanvas/.keys</code>{" "}
                  (never in the repo). An exported env var still works as
                  fallback.
                </p>
                <div className="space-y-1.5">
                  {TOP_PROVIDERS.map((id) => (
                    <CredentialRow
                      key={id}
                      providerId={id}
                      providers={providers}
                      onChanged={reload}
                    />
                  ))}
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
                    <div className="ml-1 space-y-1.5 border-l-2 border-gray-800 pl-1">
                      {moreProviders.map((id) => (
                        <CredentialRow
                          key={id}
                          providerId={id}
                          providers={providers}
                          onChanged={reload}
                        />
                      ))}
                    </div>
                  )}
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

        {/* Footer — app settings only; credentials & profiles apply instantly */}
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
            Close
          </button>
          <button
            onClick={handleSaveAppSettings}
            disabled={saving || loading}
            className={clsx(
              "flex items-center gap-1.5 rounded px-4 py-1.5 text-sm font-medium",
              saving || loading
                ? "cursor-not-allowed bg-gray-700 text-gray-500"
                : "bg-blue-600 text-white hover:bg-blue-500",
            )}
          >
            <Save size={14} />
            {saving ? "Saving..." : "Save app settings"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Credentials ── */

function CredentialRow({
  providerId,
  providers,
  onChanged,
}: {
  providerId: string;
  providers: ProvidersMap;
  onChanged: () => Promise<void>;
}) {
  const info = providers[providerId];
  const [draft, setDraft] = useState("");
  const [visible, setVisible] = useState(false);
  const [busy, setBusy] = useState(false);
  const [validation, setValidation] = useState<{
    ok: boolean;
    message: string;
  } | null>(null);

  if (!info) return null;
  const noKey = !info.key_env; // e.g. ollama

  const saveKey = async () => {
    if (!draft.trim()) return;
    setBusy(true);
    setValidation(null);
    try {
      await api.setProviderKey(providerId, draft.trim());
      setDraft("");
      window.dispatchEvent(new Event("providers-changed"));
      await onChanged();
    } finally {
      setBusy(false);
    }
  };

  const removeKey = async () => {
    setBusy(true);
    setValidation(null);
    try {
      await api.deleteProviderKey(providerId);
      window.dispatchEvent(new Event("providers-changed"));
      await onChanged();
    } finally {
      setBusy(false);
    }
  };

  const validate = async () => {
    setBusy(true);
    setValidation(null);
    try {
      const res = await api.validateProviderKey(providerId);
      setValidation(res);
    } catch {
      setValidation({ ok: false, message: "validation request failed" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span
        className="w-32 flex-shrink-0 truncate text-sm text-gray-300"
        title={`${info.label}${info.key_env ? ` — ${info.key_env}` : ""}`}
      >
        {info.label}
      </span>

      {noKey ? (
        <span className="flex-1 text-xs text-gray-500">
          no key needed — local server
        </span>
      ) : (
        <div className="relative flex-1">
          <input
            type={visible ? "text" : "password"}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && saveKey()}
            placeholder={
              info.key_source !== "none"
                ? "(key is set — paste to replace)"
                : `paste key → saved as ${info.key_env}`
            }
            className="input-field w-full pr-8 text-sm"
          />
          <button
            type="button"
            onClick={() => setVisible(!visible)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
          >
            {visible ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      )}

      {!noKey && draft.trim() && (
        <button
          onClick={saveKey}
          disabled={busy}
          className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-500"
        >
          Save
        </button>
      )}

      {/* Source badge */}
      <span
        className={clsx(
          "w-10 flex-shrink-0 text-center text-[10px] uppercase tracking-wide",
          info.key_source === "file"
            ? "text-green-400"
            : info.key_source === "env"
              ? "text-cyan-400"
              : noKey
                ? "text-gray-500"
                : "text-gray-600",
        )}
        title={
          info.key_source === "file"
            ? "from ~/.agentcanvas/.keys"
            : info.key_source === "env"
              ? `from env var ${info.key_env}`
              : "no key configured"
        }
      >
        {noKey ? "—" : info.key_source === "none" ? "no key" : info.key_source}
      </span>

      {/* Validate */}
      <button
        onClick={validate}
        disabled={busy || (!info.key_set && !noKey)}
        title="Send a minimal request to verify"
        className={clsx(
          "flex w-7 flex-shrink-0 justify-center text-gray-500",
          (info.key_set || noKey) && "hover:text-gray-200",
        )}
      >
        {busy ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <ShieldCheck size={14} />
        )}
      </button>

      {/* Remove file key */}
      <button
        onClick={removeKey}
        disabled={busy || info.key_source !== "file"}
        title="Remove key from ~/.agentcanvas/.keys"
        className={clsx(
          "flex w-6 flex-shrink-0 justify-center text-gray-600",
          info.key_source === "file" && "hover:text-red-400",
        )}
      >
        <Trash2 size={13} />
      </button>

      {validation && (
        <span
          className={clsx(
            "max-w-40 flex-shrink-0 truncate text-xs",
            validation.ok ? "text-green-400" : "text-red-400",
          )}
          title={validation.message}
        >
          {validation.ok ? "✓ valid" : `✗ ${validation.message}`}
        </span>
      )}
    </div>
  );
}

/* ── Model profiles ── */

function ProfilesSection({
  profilesState,
  providers,
  onChanged,
  onError,
}: {
  profilesState: ProfilesState;
  providers: ProvidersMap;
  onChanged: () => Promise<void>;
  onError: (msg: string) => void;
}) {
  const [adding, setAdding] = useState(false);

  const setActive = async (name: string) => {
    try {
      await api.activateProfile(name);
      await onChanged();
    } catch {
      onError("Failed to activate profile");
    }
  };

  const remove = async (name: string) => {
    try {
      await api.deleteProfile(name);
      await onChanged();
    } catch {
      onError("Failed to delete profile");
    }
  };

  const names = Object.keys(profilesState.profiles);

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-400">Model Profiles</h3>
        <button
          onClick={() => setAdding(!adding)}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200"
        >
          <Plus size={13} /> Add profile
        </button>
      </div>
      <p className="mb-2 text-xs text-gray-500">
        Named (provider, model) pairs — what LLM/VLM node dropdowns list. ★ =
        default for nodes that don't pick one.
      </p>

      {adding && (
        <ProfileEditor
          providers={providers}
          onDone={async () => {
            setAdding(false);
            await onChanged();
          }}
          onCancel={() => setAdding(false)}
          onError={onError}
        />
      )}

      <div className="space-y-1.5">
        {names.length === 0 && !adding && (
          <p className="rounded border border-gray-700 bg-gray-800/30 px-3 py-2 text-sm text-gray-500">
            No profiles yet. Add one to make a model callable.
          </p>
        )}
        {names.map((name) => {
          const p = profilesState.profiles[name];
          const isActive = profilesState.active === name;
          const provLabel = providers[p.provider]?.label || p.provider;
          return (
            <div
              key={name}
              className="flex items-center gap-2 rounded border border-gray-700 bg-gray-800/40 px-3 py-1.5"
            >
              <button
                onClick={() => setActive(name)}
                className={clsx(
                  "flex-shrink-0 transition-colors",
                  isActive
                    ? "text-yellow-400"
                    : "text-gray-600 hover:text-gray-400",
                )}
                title={isActive ? "Default profile" : "Set as default"}
              >
                <Star size={14} fill={isActive ? "currentColor" : "none"} />
              </button>
              <span
                className="w-32 flex-shrink-0 truncate text-sm font-medium text-gray-300"
                title={name}
              >
                {name}
              </span>
              <span className="flex-1 truncate text-sm text-gray-400">
                {provLabel} / {p.model}
                {p.base_url && (
                  <span className="ml-1 text-xs text-gray-600">
                    @ {p.base_url}
                  </span>
                )}
              </span>
              <span
                className={clsx(
                  "w-12 flex-shrink-0 text-right text-[10px]",
                  p.api_key_set ? "text-green-500" : "text-gray-600",
                )}
                title={
                  p.api_key_set
                    ? "provider key available"
                    : "no key for this provider — resolves to mock"
                }
              >
                {p.api_key_set ? "key ✓" : "no key"}
              </span>
              <button
                onClick={() => remove(name)}
                className="flex-shrink-0 text-gray-600 hover:text-red-400"
                title="Delete profile"
              >
                <Trash2 size={13} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ProfileEditor({
  providers,
  onDone,
  onCancel,
  onError,
}: {
  providers: ProvidersMap;
  onDone: () => Promise<void>;
  onCancel: () => void;
  onError: (msg: string) => void;
}) {
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("");
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  // Fetch live model list when the provider has a usable key
  useEffect(() => {
    setModels([]);
    const info = providers[provider];
    if (!info || (!info.key_set && provider !== "ollama")) return;
    api
      .getProviderModels(provider)
      .then((res) => setModels(res.models))
      .catch(() => {});
  }, [provider, providers]);

  const create = async () => {
    const finalModel = model || providers[provider]?.default_model || "";
    const finalName = name.trim() || finalModel;
    if (!finalName || !finalModel) return;
    setBusy(true);
    try {
      await api.createProfile({
        name: finalName,
        provider,
        model: finalModel,
        base_url: baseUrl,
      });
      await onDone();
    } catch (e) {
      onError(e instanceof Error ? e.message : "Failed to create profile");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mb-2 space-y-2 rounded border border-blue-900/60 bg-gray-800/40 p-2">
      <div className="grid grid-cols-2 gap-2">
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          className="input-field text-sm"
        >
          {Object.entries(providers).map(([id, info]) => (
            <option key={id} value={id}>
              {info.label}
              {!info.key_set && id !== "ollama" ? " (no key)" : ""}
            </option>
          ))}
        </select>
        {models.length > 0 ? (
          <select
            value={model || providers[provider]?.default_model || ""}
            onChange={(e) => setModel(e.target.value)}
            className="input-field text-sm"
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
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={`model (default: ${providers[provider]?.default_model || "—"})`}
            className="input-field text-sm"
          />
        )}
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="profile name (default: model name)"
          className="input-field text-sm"
        />
        <input
          type="text"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="base URL override (optional)"
          className="input-field text-sm"
        />
      </div>
      <div className="flex justify-end gap-2">
        <button
          onClick={onCancel}
          className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-400 hover:text-gray-200"
        >
          Cancel
        </button>
        <button
          onClick={create}
          disabled={busy}
          className="rounded bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-500"
        >
          {busy ? "Creating…" : "Create"}
        </button>
      </div>
    </div>
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
