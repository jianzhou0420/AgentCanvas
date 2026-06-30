/** PortListEditor — editable list of port entries for the "port_list" config field type. */

import { useCallback } from "react";
import { resolveInstancePorts } from "../../../../portResolution";
import { useFlowStore } from "../../../../useFlowStore";
import type { PortSchema } from "./layoutUtils";

const WIRE_TYPES = [
  "IMAGE",
  "DEPTH",
  "TEXT",
  "ACTION",
  "POSE",
  "BOOL",
  "ANY",
] as const;

export interface PortEntry {
  name: string;
  /** Canonical wire type string. May be a bare inner type (e.g. "IMAGE")
   *  or a LIST[T] wrapper (e.g. "LIST[IMAGE]"). See ADR-027. */
  wire_type: string;
  /** iterIn only: when true, the slot retains its value across iterations
   *  until next write; when false, the slot is cleared after each fire. */
  persist?: boolean;
}

/** Strip the LIST[...] wrapper, returning the inner type. */
function unwrapList(wt: string): string {
  if (wt.startsWith("LIST[") && wt.endsWith("]")) {
    return wt.slice(5, -1);
  }
  return wt;
}

function isListType(wt: string): boolean {
  return wt.startsWith("LIST[") && wt.endsWith("]");
}

interface PortListEditorProps {
  ports: PortEntry[];
  onChange: (newPorts: PortEntry[]) => void;
  /** When true, render a per-port persist checkbox (iterIn ports). */
  showPersist?: boolean;
  /** Initial persist value for new ports added via the "+ Add Port" button.
   *  iterIn init ports default to false (one-shot); iterOut defaults to true. */
  defaultPersist?: boolean;
}

export function PortListEditor({
  ports,
  onChange,
  showPersist = false,
  defaultPersist = true,
}: PortListEditorProps) {
  const addPort = useCallback(() => {
    const base: PortEntry = { name: "", wire_type: "ANY" };
    if (showPersist) base.persist = defaultPersist;
    onChange([...ports, base]);
  }, [ports, onChange, showPersist, defaultPersist]);

  const updatePort = useCallback(
    (index: number, patch: Partial<PortEntry>) => {
      onChange(ports.map((p, i) => (i === index ? { ...p, ...patch } : p)));
    },
    [ports, onChange],
  );

  const removePort = useCallback(
    (index: number) => {
      onChange(ports.filter((_, i) => i !== index));
    },
    [ports, onChange],
  );

  return (
    <div className="nopan nodrag space-y-0.5">
      {ports.map((port, i) => (
        <div key={i} className="flex items-center gap-1">
          <input
            type="text"
            value={port.name}
            placeholder="port_name"
            onChange={(e) => updatePort(i, { name: e.target.value })}
            className="min-w-0 flex-1 rounded border border-gray-700 bg-gray-800 px-1 py-0.5 text-[9px] text-gray-200"
          />
          <select
            value={unwrapList(port.wire_type)}
            onChange={(e) => {
              const next = isListType(port.wire_type)
                ? `LIST[${e.target.value}]`
                : e.target.value;
              updatePort(i, { wire_type: next });
            }}
            className="rounded border border-gray-700 bg-gray-800 px-0.5 py-0.5 text-[9px] text-gray-200"
          >
            {WIRE_TYPES.map((wt) => (
              <option key={wt} value={wt}>
                {wt}
              </option>
            ))}
          </select>
          <label
            className="flex items-center gap-0.5 text-[9px] text-gray-400"
            title="Accept multiple values; single producer auto-wrapped to length-1 list (ADR-027)"
          >
            <input
              type="checkbox"
              checked={isListType(port.wire_type)}
              onChange={(e) => {
                const inner = unwrapList(port.wire_type);
                updatePort(i, {
                  wire_type: e.target.checked ? `LIST[${inner}]` : inner,
                });
              }}
              className="h-2.5 w-2.5"
            />
            list
          </label>
          {showPersist && (
            <label
              className="flex items-center gap-0.5 text-[9px] text-gray-400"
              title="persist=ON: slot keeps value across fires until next write. persist=OFF: slot clears after each fire."
            >
              <input
                type="checkbox"
                checked={port.persist !== false}
                onChange={(e) => updatePort(i, { persist: e.target.checked })}
                className="h-2.5 w-2.5"
              />
              persist
            </label>
          )}
          <button
            onClick={() => removePort(i)}
            className="rounded px-1 py-0.5 text-[9px] text-gray-500 hover:bg-red-900/30 hover:text-red-400"
            title="Remove port"
          >
            ×
          </button>
        </div>
      ))}
      <button
        onClick={addPort}
        className="mt-0.5 w-full rounded border border-dashed border-gray-700 py-0.5 text-[9px] text-gray-500 hover:border-gray-500 hover:text-gray-300"
      >
        + Add Port
      </button>
    </div>
  );
}

/** Higher-level wrapper that wires PortListEditor into the node data + schema update + edge cleanup. */
export function PortListEditorField({
  field,
  data,
  nodeId,
  portSide,
  showPersist = false,
}: {
  field: { name: string; label: string };
  data: Record<string, unknown>;
  nodeId: string;
  /** Which schema array this field controls: "input_ports" | "output_ports" */
  portSide: "input_ports" | "output_ports";
  /** When true, render a per-port persist checkbox (iterIn ports). */
  showPersist?: boolean;
}) {
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const removeEdgesWhere = useFlowStore((s) => s.removeEdgesWhere);
  const resyncIterInPorts = useFlowStore((s) => s.resyncIterInPorts);

  const currentPorts = (data[field.name] as PortEntry[] | undefined) ?? [];
  const nodeType = (data._schema as Record<string, unknown> | undefined)
    ?.type as string | undefined;
  const isWriter = nodeType === "iterOut";
  // iterOut ports default to persist=true (loop-carried); iterIn init ports
  // pass their own defaultPersist=false via the IterInExpandPanel.
  const defaultPersist = true;

  const handleChange = useCallback(
    (newPorts: PortEntry[]) => {
      // 1. Find removed port names to clean up dangling edges
      const oldNames = new Set(currentPorts.map((p) => p.name));
      const newNames = new Set(newPorts.map((p) => p.name));
      const removed = [...oldNames].filter((n) => !newNames.has(n));

      if (removed.length > 0) {
        // iterOut source handles are the final-side names (final_<port>),
        // so dropping port X must also drop final_X edges.
        const removedSet = new Set(
          isWriter
            ? [...removed, ...removed.map((n) => `final_${n}`)]
            : removed,
        );
        removeEdgesWhere((e) => {
          if (
            e.source === nodeId &&
            e.sourceHandle &&
            removedSet.has(e.sourceHandle)
          )
            return true;
          if (
            e.target === nodeId &&
            e.targetHandle &&
            removedSet.has(e.targetHandle)
          )
            return true;
          return false;
        });
      }

      // 2. Build updated _schema with new ports list. iterOut goes through
      // resolveInstancePorts so the derived surfaces (stop input, final_*
      // outputs) are rebuilt rather than clobbered.
      const schema = data._schema as Record<string, unknown> | undefined;
      const updatedSchema = schema
        ? isWriter
          ? (resolveInstancePorts(schema, { ports: newPorts }) as Record<
              string,
              unknown
            >)
          : {
              ...schema,
              [portSide]: newPorts.map(
                (p): PortSchema => ({
                  name: p.name,
                  wire_type: p.wire_type,
                  description: "",
                  optional: false,
                  ...(p.persist !== undefined ? { persist: p.persist } : {}),
                }),
              ),
            }
        : undefined;

      // 3. Persist to node data (triggers React Flow re-render → handle re-layout)
      const patch: Record<string, unknown> = { [field.name]: newPorts };
      if (updatedSchema) patch._schema = updatedSchema;
      updateNodeData(nodeId, patch);

      // 4. If this is an iterOut writer, resync the paired iterIn so its
      //    output handles track the writer's new port surface live.
      if (isWriter) resyncIterInPorts();
    },
    [
      currentPorts,
      data,
      field.name,
      nodeId,
      portSide,
      isWriter,
      removeEdgesWhere,
      updateNodeData,
      resyncIterInPorts,
    ],
  );

  return (
    <div className="nopan nodrag block text-[9px] text-gray-400">
      <div className="mb-0.5">{field.label}</div>
      <PortListEditor
        ports={currentPorts}
        onChange={handleChange}
        showPersist={showPersist}
        defaultPersist={defaultPersist}
      />
    </div>
  );
}
