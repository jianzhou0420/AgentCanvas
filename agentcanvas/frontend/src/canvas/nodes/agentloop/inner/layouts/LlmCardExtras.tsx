/** LLM-node card decorations — the "gauge, not a form" layer.
 *
 * Model chip (↪ = Default mode, red = resolves to mock),
 * read-only template preview with {var} tinting, and a summary line of the
 * explicitly-set sampling params. All data comes from node config + the
 * server-injected profile options — nothing is editable here; editing lives
 * in the properties panel.
 */

import type { NodeSchema } from "./layoutUtils";
import { useFlowStore } from "../../../../useFlowStore";
import { useNodeOutput } from "./useNodeOutput";
import { providerByLabel, useProviders } from "../../../../providersCache";

export interface ModelRef {
  mode: "default" | "profile" | "direct";
  /** Chip text (without the ↪ prefix). */
  display: string;
  /** True when resolution is known to fall back to mock. */
  mock: boolean;
}

/** Resolve the node's model reference for display. Named/default modes read
 * the server-injected profile options (labels like "Default (OpenAI /
 * gpt-5-nano)" / "OpenAI / gpt-5-nano"); Direct mode reads inline config. */
export function resolveModelRef(
  data: Record<string, unknown>,
  schema: NodeSchema | undefined,
): ModelRef {
  const profile = String(data.profile ?? "");

  if (profile === "__direct__") {
    const provider = String(data.provider ?? "");
    const model = String(data.model ?? "");
    if (!provider || !model) {
      return { mode: "direct", display: "incomplete model pick", mock: true };
    }
    return { mode: "direct", display: `${model} · ${provider}`, mock: false };
  }

  const options =
    schema?.ui_config?.config_fields?.find((f) => f.name === "profile")
      ?.options ?? [];

  if (profile === "") {
    const label = options.find((o) => o.value === "")?.label ?? "Default";
    const inner = label.replace(/^Default\s*\(/, "").replace(/\)$/, "");
    if (inner === "none" || inner === "Default") {
      return { mode: "default", display: "no default model", mock: true };
    }
    return { mode: "default", display: inner, mock: false };
  }

  const opt = options.find((o) => o.value === profile);
  if (!opt) {
    // Profile referenced by the graph but no longer in the store.
    return { mode: "profile", display: `${profile} (missing)`, mock: true };
  }
  return { mode: "profile", display: opt.label, mock: false };
}

/** The always-visible model chip. */
export function LlmModelChip({
  data,
  schema,
}: {
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}) {
  const providers = useProviders();
  const ref = resolveModelRef(data, schema);

  // Resolution can succeed while the provider still has no usable key —
  // the call would only surface as a mock response at run time. Check
  // key status here so the warning exists at edit time.
  let noKey = false;
  if (!ref.mock && providers) {
    const info =
      ref.mode === "direct"
        ? providers[String(data.provider ?? "")]
        : providerByLabel(providers, ref.display.split(" / ")[0].trim())?.[1];
    noKey = !!info && !info.key_set;
  }

  const chipClass = ref.mock
    ? "border-red-500/70 bg-red-950/40 text-red-300"
    : noKey
      ? "border-amber-500/70 bg-amber-950/40 text-amber-300"
      : "border-gray-600 bg-gray-800/70 text-gray-300";

  return (
    <div className="mt-0.5 flex justify-center">
      <span
        className={`rounded-full border px-2 py-px font-mono text-[8px] ${chipClass}`}
        title={
          ref.mock
            ? "Resolution fails — this node will return mock responses"
            : noKey
              ? "Provider has no API key — the node will return mock responses (Settings → Credentials)"
              : ref.mode === "default"
                ? "Follows the active profile (Settings → Model Profiles)"
                : ref.mode === "direct"
                  ? "Model pinned directly on this node"
                  : "Named profile"
        }
      >
        {ref.mock ? "⚠ mock — " : noKey ? "⚠ no key — " : ""}
        {!ref.mock && !noKey && ref.mode === "default" ? "↪ " : ""}
        {ref.display}
      </span>
    </div>
  );
}

function fmtTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

/** Last-call gauge line — tokens in→out, wall-clock, cost. Fed by the
 * executor's per-node llm_usage WS event; empty until the node fires. */
export function LlmUsageLine({ nodeId }: { nodeId: string }) {
  const usage = useNodeOutput(nodeId)?.usage;
  if (!usage || !usage.calls) return null;

  const bits = [
    `${fmtTokens(usage.prompt_tokens)}→${fmtTokens(usage.completion_tokens)} tok`,
  ];
  if (usage.duration_ms !== undefined) {
    bits.push(
      usage.duration_ms >= 1000
        ? `${(usage.duration_ms / 1000).toFixed(1)}s`
        : `${Math.round(usage.duration_ms)}ms`,
    );
  }
  if (usage.usd_cost > 0) bits.push(`$${usage.usd_cost.toFixed(4)}`);
  if (usage.calls > 1) bits.push(`×${usage.calls}`);

  return (
    <div
      className="mx-1 mt-1 border-t border-gray-800 pt-0.5 text-center font-mono text-[7px] text-emerald-500/80"
      title={`last firing — model ${usage.model || "?"}, ${usage.total_tokens} total tokens${
        usage.cached_tokens ? ` (${usage.cached_tokens} cached)` : ""
      }`}
    >
      {bits.join(" · ")}
    </div>
  );
}

/** Read-only first-lines preview of the prompt template. Three-way {var}
 * tinting mirrors the runtime resolution order: input port (sky) →
 * graph_state key (violet, needs an access grant) → unresolved (amber —
 * renders as "(no var)" at run time). */
export function LlmTemplatePreview({
  data,
  schema,
}: {
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}) {
  const graphState = useFlowStore((s) => s.graphState);
  const template = String(data.template ?? "");
  if (!template.trim()) return null;

  const portsCfg = data.ports as Array<{ name: string }> | undefined;
  const portNames = new Set(
    (portsCfg ?? schema?.input_ports ?? []).map((p) => p.name),
  );
  const stateKeys = new Set(Object.keys(graphState?.states ?? {}));

  const excerpt = template.slice(0, 110);
  const parts = excerpt.split(/(\{\w+\})/g);

  return (
    <div
      className="mx-1 mt-1 truncate whitespace-pre-line break-words border-l border-gray-700 pl-1 text-left font-mono text-[7px] leading-tight text-gray-500"
      style={{ maxHeight: 26, overflow: "hidden" }}
      title={template}
    >
      {parts.map((part, i) => {
        const m = part.match(/^\{(\w+)\}$/);
        if (!m) return <span key={i}>{part}</span>;
        const name = m[1];
        if (portNames.has(name) || name === "step") {
          return (
            <span key={i} className="text-sky-400" title="fills from input port">
              {part}
            </span>
          );
        }
        if (stateKeys.has(name)) {
          return (
            <span
              key={i}
              className="text-violet-400"
              title="fills from graph_state — needs an access grant on this node, else the firing fails"
            >
              {part}
            </span>
          );
        }
        return (
          <span
            key={i}
            className="text-amber-400"
            title={`no port or state key named "${name}" — the firing will fail with a template error`}
          >
            {part}
          </span>
        );
      })}
      {template.length > excerpt.length ? "…" : ""}
    </div>
  );
}

/** Tiny uppercase section header — card-scale echo of the properties
 * panel's group titles ("Model & Sampling", "Prompt", …). */
export function CardSectionHeader({ children }: { children: string }) {
  return (
    <div className="mx-1 mt-1.5 border-b border-gray-800 pb-px text-left text-[7px] uppercase tracking-wider text-gray-600">
      {children}
    </div>
  );
}

/** Labeled sampling rows under a "Model & Sampling" header — the same
 * grouping as the properties panel, rendered read-only at card scale.
 * Unset params show "provider default" (the two-state contract). */
export function LlmSamplingBlock({
  data,
  schema,
}: {
  data: Record<string, unknown>;
  schema: NodeSchema | undefined;
}) {
  const isUnset = (v: unknown) => v === null || v === undefined || v === "";

  const rows: Array<{ label: string; value: string; unset: boolean }> = [
    {
      label: "temperature",
      value: isUnset(data.temperature)
        ? "provider default"
        : String(data.temperature),
      unset: isUnset(data.temperature),
    },
    {
      label: "max tokens",
      value: isUnset(data.max_tokens)
        ? "provider default"
        : String(data.max_tokens),
      unset: isUnset(data.max_tokens),
    },
  ];
  const n = Number(data.n ?? 1);
  if (n > 1) rows.push({ label: "n", value: String(n), unset: false });

  // Image detail is operative whenever the call ships images (VLM path):
  // show the explicit value, or the backend's effective default in italics.
  const ports = (data.ports ?? schema?.input_ports ?? []) as Array<{
    wire_type?: string;
  }>;
  const consumesImages = ports.some((p) =>
    String(p.wire_type ?? "").includes("IMAGE"),
  );
  const detail = String(data.image_detail ?? "");
  if (detail) {
    rows.push({ label: "image detail", value: detail, unset: false });
  } else if (consumesImages) {
    const schemaDefault = String(
      schema?.ui_config?.config_fields?.find((f) => f.name === "image_detail")
        ?.default ?? "low",
    );
    rows.push({ label: "image detail", value: schemaDefault, unset: true });
  }

  return (
    <div>
      <CardSectionHeader>Model & Sampling</CardSectionHeader>
      <div className="mx-1 mt-0.5 space-y-px">
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-baseline justify-between gap-2 text-[7px] leading-tight"
          >
            <span className="text-gray-600">{r.label}</span>
            <span
              className={
                r.unset ? "italic text-gray-600" : "font-mono text-gray-400"
              }
            >
              {r.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
