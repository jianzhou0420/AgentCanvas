/** Properties panel — shows and edits the selected node's properties.
 *
 * Displays node type, label, position, and all data/config fields.
 * Editable fields update the node in the React Flow store.
 * Includes a dedicated "Hooks" section for per-node hook configuration.
 */

import { useCallback } from "react";
import { Settings, MapPin, Tag, Box, Zap, Plus, Trash2 } from "lucide-react";
import { useFlowStore } from "../useFlowStore";
import type { HookDef } from "../types";
import { ConfigFieldRenderer } from "../nodes/agentloop/inner/layouts/ConfigFieldRenderer";
import { IterInExpandPanel } from "../nodes/agentloop/inner/layouts/IterInExpandPanel";
import {
  CapAnnotatedField,
  ModelRefPicker,
  useModelCapabilities,
} from "./LlmModelControls";
import type {
  ConfigFieldSchema,
  UIConfigSchema,
  PortSchema,
} from "../nodes/agentloop/inner/layouts/layoutUtils";

/** Determine if a value is a simple editable type. */
function isEditable(v: unknown): v is string | number | boolean {
  return (
    typeof v === "string" || typeof v === "number" || typeof v === "boolean"
  );
}

/** Keys to skip in the properties list (internal/non-user-facing). */
const SKIP_KEYS = new Set([
  "_schema",
  "subgraph",
  "innerGraph",
  "_preview",
  "label", // shown separately in header
  "hooks", // rendered in dedicated HooksSection
]);

/** Panel section titles for ConfigField.section groups. */
const SECTION_TITLES: Record<string, string> = {
  model: "Model & Sampling",
  prompt: "Prompt",
  wiring: "Wiring",
  "": "Config",
};
const SECTION_ORDER = ["model", "prompt", "wiring", ""];

export default function PropertiesPanel() {
  const selectedNodeId = useFlowStore((s) => s.selectedNodeId);
  const visibleNodes = useFlowStore((s) => s.visibleNodes);
  const node = visibleNodes.find((n) => n.id === selectedNodeId);

  // llm nodes: fetch rulebook verdicts for the resolved (provider, model).
  // Hook must run unconditionally — enabled=false when no llm node selected.
  const nodeDataForCaps = (node?.data ?? {}) as Record<string, unknown>;
  const nodeSchemaForCaps = nodeDataForCaps._schema as
    | Record<string, unknown>
    | undefined;
  const isLlmNode =
    ((nodeSchemaForCaps?.category as string) || "") === "llm";
  const caps = useModelCapabilities(nodeDataForCaps, isLlmNode);

  if (!node) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-xs text-gray-600">
          Select a node to view properties
        </span>
      </div>
    );
  }

  const data = (node.data || {}) as Record<string, unknown>;
  const schema = data._schema as Record<string, unknown> | undefined;
  const label =
    (data.label as string) ||
    (schema?.display_name as string) ||
    node.type ||
    "—";
  const category = (schema?.category as string) || "—";
  const description = (schema?.description as string) || "";

  const uiConfig = schema?.ui_config as UIConfigSchema | undefined;
  const configFields: ConfigFieldSchema[] = uiConfig?.config_fields ?? [];
  const configFieldNames = new Set(configFields.map((f) => f.name));

  // Collect editable fields from data (excluding internal keys + ones owned by a ConfigField)
  const fields = Object.entries(data).filter(
    ([key]) =>
      !SKIP_KEYS.has(key) &&
      !key.startsWith("_") &&
      !configFieldNames.has(key) &&
      // Direct-mode model reference is owned by the ModelRefPicker
      !(isLlmNode && (key === "provider" || key === "model")),
  );

  // Group config fields by section (llm nodes declare model/prompt/wiring;
  // nodes without sections fall into the single "Config" group as before).
  const sectionedFields = new Map<string, ConfigFieldSchema[]>();
  for (const f of configFields) {
    const section = f.section && SECTION_TITLES[f.section] ? f.section : "";
    if (!sectionedFields.has(section)) sectionedFields.set(section, []);
    sectionedFields.get(section)!.push(f);
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-800 px-3 py-1.5">
        <div className="flex items-center gap-2">
          <Settings size={12} className="text-blue-400" />
          <span className="text-xs font-semibold text-blue-300">{label}</span>
          <span className="rounded bg-gray-800 px-1.5 py-0.5 text-[9px] text-gray-500">
            {node.type}
          </span>
        </div>
        {description && (
          <div className="mt-0.5 text-[10px] text-gray-500">{description}</div>
        )}
      </div>

      {/* Properties */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-left text-[10px] uppercase tracking-wider text-gray-600">
              <th className="px-3 py-1 w-1/3">Property</th>
              <th className="px-3 py-1">Value</th>
            </tr>
          </thead>
          <tbody>
            {/* Identity */}
            <PropertyRow
              icon={<Tag size={10} />}
              label="id"
              value={node.id}
              readonly
            />
            <PropertyRow
              icon={<Box size={10} />}
              label="type"
              value={node.type || "—"}
              readonly
            />
            <PropertyRow label="category" value={category} readonly />
            <PropertyRow
              icon={<MapPin size={10} />}
              label="position"
              value={`${Math.round(node.position.x)}, ${Math.round(node.position.y)}`}
              readonly
            />

            {/* Editable label */}
            <EditableRow
              nodeId={node.id}
              label="label"
              value={(data.label as string) || ""}
            />

            {/* Data fields */}
            {fields.map(([key, value]) => {
              if (isEditable(value)) {
                return (
                  <EditableRow
                    key={key}
                    nodeId={node.id}
                    label={key}
                    value={value}
                  />
                );
              }
              // Non-editable complex values
              const display =
                value === null
                  ? "null"
                  : value === undefined
                    ? "—"
                    : Array.isArray(value)
                      ? `[${value.length} items]`
                      : typeof value === "object"
                        ? "{...}"
                        : String(value);
              return (
                <PropertyRow key={key} label={key} value={display} readonly />
              );
            })}
          </tbody>
        </table>

        {/* Declarative config fields, grouped by section */}
        {SECTION_ORDER.filter((s) => sectionedFields.has(s)).map((section) => (
          <div key={section} className="border-t border-gray-800 px-3 py-2">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-gray-600">
              {SECTION_TITLES[section]}
            </div>
            <div className="space-y-1.5">
              {sectionedFields.get(section)!.map((field) => {
                // llm model reference → three-mode picker (Default /
                // named profile / Browse provider→model pinned inline)
                if (isLlmNode && field.name === "profile") {
                  return (
                    <ModelRefPicker
                      key={field.name}
                      field={field}
                      data={data}
                      nodeId={node.id}
                    />
                  );
                }
                if (isLlmNode) {
                  return (
                    <CapAnnotatedField
                      key={field.name}
                      field={field}
                      data={data}
                      nodeId={node.id}
                      caps={caps}
                    />
                  );
                }
                return (
                  <ConfigFieldRenderer
                    key={field.name}
                    field={field}
                    data={data}
                    nodeId={node.id}
                  />
                );
              })}
            </div>
          </div>
        ))}

        {/* iterIn-specific: synthesised port list + persist toggles */}
        {node.type === "iterIn" && (
          <div className="border-t border-gray-800 px-3 py-2">
            <div className="mb-1 text-[10px] uppercase tracking-wider text-gray-600">
              Iteration Ports
            </div>
            <IterInExpandPanel
              iterInId={node.id}
              ports={
                (((data._schema as Record<string, unknown> | undefined)
                  ?.output_ports as PortSchema[] | undefined) ??
                  []) as unknown as Array<{
                  name: string;
                  wire_type: string;
                  persist?: boolean;
                  origin?: "init" | "iterOut";
                  writer_name?: string;
                }>
              }
            />
          </div>
        )}

        {/* Hooks section */}
        <HooksSection
          nodeId={node.id}
          hooks={(data.hooks as HookDef[]) || []}
        />
      </div>
    </div>
  );
}

/* ── Read-only row ── */

function PropertyRow({
  icon,
  label,
  value,
  readonly: _readonly,
}: {
  icon?: React.ReactNode;
  label: string;
  value: string;
  readonly?: boolean;
}) {
  return (
    <tr className="border-b border-gray-800/50">
      <td className="px-3 py-1 text-gray-500">
        <span className="flex items-center gap-1">
          {icon}
          {label}
        </span>
      </td>
      <td className="px-3 py-1 font-mono text-gray-400">{value}</td>
    </tr>
  );
}

/* ── Editable row ── */

function EditableRow({
  nodeId,
  label,
  value,
}: {
  nodeId: string;
  label: string;
  value: string | number | boolean;
}) {
  const handleChange = useCallback(
    (newValue: string | number | boolean) => {
      useFlowStore.getState().updateNodeData(nodeId, { [label]: newValue });
    },
    [nodeId, label],
  );

  if (typeof value === "boolean") {
    return (
      <tr className="border-b border-gray-800/50 hover:bg-gray-800/30">
        <td className="px-3 py-1 text-gray-400">{label}</td>
        <td className="px-3 py-1">
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => handleChange(e.target.checked)}
            className="h-3 w-3"
          />
        </td>
      </tr>
    );
  }

  if (typeof value === "number") {
    return (
      <tr className="border-b border-gray-800/50 hover:bg-gray-800/30">
        <td className="px-3 py-1 text-gray-400">{label}</td>
        <td className="px-3 py-1">
          <input
            type="number"
            value={value}
            onChange={(e) => handleChange(Number(e.target.value))}
            className="w-full rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 font-mono text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
          />
        </td>
      </tr>
    );
  }

  // String
  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30">
      <td className="px-3 py-1 text-gray-400">{label}</td>
      <td className="px-3 py-1">
        <input
          type="text"
          value={value}
          onChange={(e) => handleChange(e.target.value)}
          className="w-full rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 font-mono text-xs text-gray-200 focus:border-blue-500 focus:outline-none"
        />
      </td>
    </tr>
  );
}

/* ── Hooks section ── */

const NODE_HOOK_EVENTS: HookDef["event"][] = [
  "PreNodeExecute",
  "PostNodeExecute",
];

function HooksSection({ nodeId, hooks }: { nodeId: string; hooks: HookDef[] }) {
  const updateHooks = useCallback(
    (newHooks: HookDef[]) => {
      useFlowStore.getState().updateNodeData(nodeId, { hooks: newHooks });
    },
    [nodeId],
  );

  const addHook = useCallback(() => {
    updateHooks([
      ...hooks,
      {
        event: "PostNodeExecute",
        command: "",
        timeout_ms: 1000,
        enabled: true,
      },
    ]);
  }, [hooks, updateHooks]);

  const removeHook = useCallback(
    (index: number) => updateHooks(hooks.filter((_, i) => i !== index)),
    [hooks, updateHooks],
  );

  const updateHook = useCallback(
    (index: number, patch: Partial<HookDef>) =>
      updateHooks(hooks.map((h, i) => (i === index ? { ...h, ...patch } : h))),
    [hooks, updateHooks],
  );

  return (
    <div className="border-t border-gray-800 px-3 py-2">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-gray-600">
          <Zap size={10} />
          Hooks
        </span>
        <button
          onClick={addHook}
          className="flex items-center gap-0.5 rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400 hover:bg-gray-700 hover:text-gray-200"
        >
          <Plus size={9} />
          Add
        </button>
      </div>
      {hooks.length === 0 && (
        <div className="text-[10px] text-gray-600">No hooks configured</div>
      )}
      {hooks.map((hook, i) => (
        <HookRow
          key={i}
          hook={hook}
          onChange={(patch) => updateHook(i, patch)}
          onRemove={() => removeHook(i)}
        />
      ))}
    </div>
  );
}

function HookRow({
  hook,
  onChange,
  onRemove,
}: {
  hook: HookDef;
  onChange: (patch: Partial<HookDef>) => void;
  onRemove: () => void;
}) {
  const inputCls =
    "w-full rounded border border-gray-700 bg-gray-800 px-1.5 py-0.5 font-mono text-xs text-gray-200 focus:border-blue-500 focus:outline-none";

  return (
    <div className="mb-1.5 rounded border border-gray-800 bg-gray-900/50 p-1.5">
      <div className="mb-1 flex items-center gap-1">
        {/* Event selector */}
        <select
          value={hook.event}
          onChange={(e) =>
            onChange({ event: e.target.value as HookDef["event"] })
          }
          className="flex-1 rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[10px] text-gray-200 focus:border-blue-500 focus:outline-none"
        >
          {NODE_HOOK_EVENTS.map((ev) => (
            <option key={ev} value={ev}>
              {ev}
            </option>
          ))}
        </select>
        {/* Enabled toggle */}
        <input
          type="checkbox"
          checked={hook.enabled !== false}
          onChange={(e) => onChange({ enabled: e.target.checked })}
          className="h-3 w-3"
          title="Enabled"
        />
        {/* Remove */}
        <button
          onClick={onRemove}
          className="rounded p-0.5 text-gray-600 hover:bg-red-900/30 hover:text-red-400"
          title="Remove hook"
        >
          <Trash2 size={10} />
        </button>
      </div>
      {/* Command */}
      <input
        type="text"
        value={hook.command}
        onChange={(e) => onChange({ command: e.target.value })}
        placeholder="command (e.g. python hook.py)"
        className={inputCls}
      />
      {/* Timeout */}
      <div className="mt-1 flex items-center gap-1">
        <span className="text-[9px] text-gray-600">timeout</span>
        <input
          type="number"
          value={hook.timeout_ms ?? 1000}
          onChange={(e) => onChange({ timeout_ms: Number(e.target.value) })}
          className="w-16 rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[10px] text-gray-200 focus:border-blue-500 focus:outline-none"
        />
        <span className="text-[9px] text-gray-600">ms</span>
      </div>
    </div>
  );
}
