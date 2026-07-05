/** Inline config field controls — renders a single ConfigFieldSchema as a UI widget. */

import { useCallback } from "react";
import type { ConfigFieldSchema } from "./layoutUtils";
import { PortListEditorField } from "./PortListEditor";
import { useFlowStore } from "../../../../useFlowStore";

export function ConfigFieldRenderer({
  field,
  data,
  nodeId,
}: {
  field: ConfigFieldSchema;
  data: Record<string, unknown>;
  nodeId: string;
}) {
  const updateNodeData = useFlowStore((s) => s.updateNodeData);

  const updateData = useCallback(
    (key: string, value: unknown) => {
      updateNodeData(nodeId, { [key]: value });
    },
    [nodeId, updateNodeData],
  );

  const currentValue = data[field.name] ?? field.default;

  switch (field.field_type) {
    case "label":
      return (
        <div className="text-center text-[9px] text-gray-400">
          {field.label}: {String(currentValue ?? "")}
        </div>
      );

    case "slider": {
      // Two-state fields (unset_label set): unset = value never sent, the
      // provider's own default applies. A click arms the slider; ✕ unsets.
      const isUnset =
        currentValue === null || currentValue === undefined || currentValue === "";
      if (field.unset_label && isUnset) {
        return (
          <div className="nopan nodrag flex items-center justify-between text-[9px] text-gray-400">
            <span>{field.label}</span>
            <button
              type="button"
              onClick={() =>
                updateData(field.name, field.min ?? 0)
              }
              className="rounded border border-dashed border-gray-600 px-1.5 py-0.5 text-[9px] italic text-gray-500 hover:border-gray-400 hover:text-gray-300"
              title="Unset — the provider default applies. Click to set a value."
            >
              {field.unset_label}
            </button>
          </div>
        );
      }
      const num = Number(currentValue ?? field.min ?? 0);
      return (
        <label className="nopan nodrag block text-[9px] text-gray-400">
          <span className="flex items-center justify-between">
            <span>
              {field.label}: {num.toFixed(1)}
            </span>
            {field.unset_label && (
              <button
                type="button"
                onClick={() => updateData(field.name, null)}
                className="text-gray-600 hover:text-gray-300"
                title={`Unset (back to ${field.unset_label})`}
              >
                ✕
              </button>
            )}
          </span>
          <input
            type="range"
            min={field.min ?? 0}
            max={field.max ?? 1}
            step={field.step ?? 0.1}
            value={num}
            onChange={(e) => updateData(field.name, Number(e.target.value))}
            className="mt-0.5 w-full"
          />
        </label>
      );
    }

    case "text":
      return (
        <label className="nopan nodrag block text-[9px] text-gray-400">
          {field.label}
          <input
            type="text"
            value={String(currentValue ?? "")}
            placeholder={field.placeholder}
            onChange={(e) => updateData(field.name, e.target.value)}
            className="mt-0.5 w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
          />
        </label>
      );

    case "select":
      return (
        <label className="nopan nodrag block text-[9px] text-gray-400">
          {field.label}
          <select
            value={String(currentValue ?? "")}
            onChange={(e) => updateData(field.name, e.target.value)}
            className="mt-0.5 w-full rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
          >
            {(field.options || []).map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      );

    case "toggle": {
      const checked = Boolean(currentValue);
      return (
        <label className="nopan nodrag flex items-center gap-1 text-[9px] text-gray-400">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => updateData(field.name, e.target.checked)}
            className="h-3 w-3"
          />
          {field.label}
        </label>
      );
    }

    case "textarea":
      return (
        <label className="nopan nodrag nowheel block text-[9px] text-gray-400">
          {field.label}
          <textarea
            value={String(currentValue ?? "")}
            placeholder={field.placeholder}
            rows={3}
            onChange={(e) => updateData(field.name, e.target.value)}
            className="mt-0.5 w-full resize-none rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
          />
        </label>
      );

    case "port_list": {
      const portSide =
        field.port_side === "output_ports" ? "output_ports" : "input_ports";
      return (
        <PortListEditorField
          field={field}
          data={data}
          nodeId={nodeId}
          portSide={portSide}
          showPersist={field.show_persist_toggle === true}
        />
      );
    }

    default:
      return null;
  }
}
