/** Graph-level state banner — fixed bar above the canvas showing the graph_state container.
 *
 * Not a React Flow node — rendered as a regular React component above
 * the canvas. Shows all named states with live value previews from WS.
 * Collapsed to a subtle link when no graph_state is set.
 */

import { useState, useCallback, useMemo } from "react";
import { Database, Plus, X, Settings } from "lucide-react";
import clsx from "clsx";
import { useFlowStore } from "../useFlowStore";
import type { ContainerDef, StateDef } from "../types";

const STATE_TYPE_LABELS: Record<string, string> = {
  accumulator: "acc",
  lastWrite: "lw",
  counter: "cnt",
  ephemeral: "eph",
};

const STATE_TYPE_COLORS: Record<string, string> = {
  accumulator: "text-amber-400",
  lastWrite: "text-blue-400",
  counter: "text-green-400",
  ephemeral: "text-gray-400",
};

/* ── Add State Modal ── */

function AddStateModal({
  onAdd,
  onClose,
}: {
  onAdd: (name: string, type: string, valueType: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [type, setType] = useState("accumulator");
  const [valueType, setValueType] = useState("TEXT");

  const handleSubmit = () => {
    if (!name.trim()) return;
    onAdd(name.trim(), type, valueType);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-80 rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl">
        <h3 className="mb-3 text-sm font-semibold text-gray-200">Add State</h3>
        <div className="space-y-2">
          <input
            type="text"
            placeholder="State name (e.g. action_history)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200 focus:border-violet-500 focus:outline-none"
            autoFocus
          />
          <select
            value={type}
            onChange={(e) => setType(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200"
          >
            <option value="accumulator">Accumulator (list.append)</option>
            <option value="lastWrite">Last Write (overwrite)</option>
            <option value="counter">Counter (sum)</option>
            <option value="ephemeral">Ephemeral (clears per step)</option>
          </select>
          <input
            type="text"
            placeholder="Value type (TEXT, ACTION, IMAGE, ANY...)"
            value={valueType}
            onChange={(e) => setValueType(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200"
          />
        </div>
        <div className="mt-3 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded px-3 py-1 text-xs text-gray-400 hover:bg-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!name.trim()}
            className="rounded bg-violet-600 px-3 py-1 text-xs text-white hover:bg-violet-500 disabled:bg-gray-700 disabled:text-gray-500"
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Main Banner ── */

export default function GraphStateBanner() {
  const canvasStack = useFlowStore((s) => s.canvasStack);
  const nodes = useFlowStore((s) => s.nodes);
  const rootGraphState = useFlowStore((s) => s.graphState);
  const preview = useFlowStore((s) => s.graphStatePreview);
  const setGraphState = useFlowStore((s) => s.setGraphState);

  // Stack-aware: inside a composite, show its subgraph's "graph_state"
  // container (the well-known id used after ADR-026).
  const graphState = useMemo(() => {
    if (canvasStack.length === 0) return rootGraphState;
    const compositeId = canvasStack[canvasStack.length - 1].nodeId;
    const compositeNode = nodes.find((n) => n.id === compositeId);
    const subgraph = compositeNode?.data?.subgraph as
      | import("../types").GraphDefinition
      | undefined;
    const gs = (subgraph?.containers || []).find((c) => c.id === "graph_state");
    return (gs as import("../types").ContainerDef | undefined) ?? null;
  }, [canvasStack, nodes, rootGraphState]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingLabel, setEditingLabel] = useState(false);
  const [labelDraft, setLabelDraft] = useState("");

  const handleCreate = useCallback(() => {
    setGraphState({
      id: `gs_${Date.now().toString(36)}`,
      label: "Graph State",
      position: { x: 0, y: 0 },
      states: {},
    });
  }, [setGraphState]);

  const handleRemove = useCallback(() => {
    setGraphState(null);
  }, [setGraphState]);

  const handleAddState = useCallback(
    (name: string, type: string, valueType: string) => {
      if (!graphState) return;
      const newState: StateDef = {
        type: type as StateDef["type"],
        value_type: valueType,
      };
      setGraphState({
        ...graphState,
        states: { ...graphState.states, [name]: newState },
      });
    },
    [graphState, setGraphState],
  );

  const handleRemoveState = useCallback(
    (name: string) => {
      if (!graphState) return;
      const { [name]: _, ...rest } = graphState.states;
      setGraphState({ ...graphState, states: rest });
    },
    [graphState, setGraphState],
  );

  const handleLabelSave = useCallback(() => {
    if (!graphState || !labelDraft.trim()) return;
    setGraphState({ ...graphState, label: labelDraft.trim() });
    setEditingLabel(false);
  }, [graphState, labelDraft, setGraphState]);

  // Collapsed: no graph_state set
  if (!graphState) {
    return (
      <div className="flex items-center border-b border-gray-800/50 bg-gray-900/50 px-3 py-0.5">
        <button
          onClick={handleCreate}
          className="flex items-center gap-1 text-[10px] text-gray-600 transition hover:text-amber-400"
        >
          <Plus size={10} />
          Add Graph State
        </button>
      </div>
    );
  }

  const stateEntries = Object.entries(graphState.states);

  return (
    <>
      <div className="flex items-center gap-2 border-b border-amber-500/20 bg-amber-950/10 px-3 py-1">
        {/* Icon + label */}
        <Database size={12} className="shrink-0 text-amber-400" />
        {editingLabel ? (
          <input
            type="text"
            value={labelDraft}
            onChange={(e) => setLabelDraft(e.target.value)}
            onBlur={handleLabelSave}
            onKeyDown={(e) => e.key === "Enter" && handleLabelSave()}
            className="w-32 rounded border border-amber-500/30 bg-transparent px-1 text-[11px] font-medium text-amber-300 focus:outline-none"
            autoFocus
          />
        ) : (
          <span
            className="cursor-pointer text-[11px] font-medium text-amber-300 hover:text-amber-200"
            onClick={() => {
              setLabelDraft(graphState.label);
              setEditingLabel(true);
            }}
          >
            {graphState.label || "Graph State"}
          </span>
        )}

        {/* State entries */}
        <div className="flex flex-1 items-center gap-1.5 overflow-x-auto">
          {stateEntries.map(([name, entry]) => {
            const liveData = preview?.[name] as
              | Record<string, unknown>
              | undefined;
            return (
              <div
                key={name}
                className="group flex shrink-0 items-center gap-1 rounded bg-gray-800/40 px-1.5 py-0.5"
              >
                <span className="text-[9px] text-amber-400">●</span>
                <span className="text-[10px] text-gray-300">{name}</span>
                <span
                  className={clsx(
                    "text-[8px]",
                    STATE_TYPE_COLORS[entry.type] || "text-gray-500",
                  )}
                >
                  {STATE_TYPE_LABELS[entry.type] || entry.type}
                </span>
                <span className="text-[8px] text-gray-600">
                  {entry.value_type}
                </span>
                {liveData && (
                  <span className="text-[8px] text-gray-500">
                    {liveData.size !== undefined ? `${liveData.size}` : ""}
                    {liveData.value !== undefined ? `${liveData.value}` : ""}
                    {!liveData.size &&
                    liveData.value === undefined &&
                    liveData.preview
                      ? String(liveData.preview).slice(0, 20)
                      : ""}
                  </span>
                )}
                <button
                  onClick={() => handleRemoveState(name)}
                  className="hidden text-gray-600 hover:text-red-400 group-hover:block"
                >
                  <X size={8} />
                </button>
              </div>
            );
          })}
        </div>

        {/* Actions */}
        <button
          onClick={() => setShowAddModal(true)}
          className="rounded p-0.5 text-gray-500 transition hover:bg-gray-800 hover:text-amber-400"
          title="Add state"
        >
          <Plus size={12} />
        </button>
        <button
          onClick={handleRemove}
          className="rounded p-0.5 text-gray-500 transition hover:bg-gray-800 hover:text-red-400"
          title="Remove graph state"
        >
          <X size={12} />
        </button>
      </div>

      {showAddModal && (
        <AddStateModal
          onAdd={handleAddState}
          onClose={() => setShowAddModal(false)}
        />
      )}
    </>
  );
}
