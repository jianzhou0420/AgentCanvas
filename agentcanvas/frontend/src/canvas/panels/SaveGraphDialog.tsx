/** Save graph dialog — modal for saving the current graph to workspace/graphs/. */

import { useState, useEffect } from "react";
import { Save, X } from "lucide-react";
import { api } from "../../api";
import type { SavedGraph } from "../../types";
import { useFlowStore, mergeGraphState } from "../useFlowStore";
import { flowToGraph } from "../graphConversion";

interface SaveGraphDialogProps {
  onClose: () => void;
  onSaved: () => void;
  /** "graph" = editable template, "node" = frozen composite archive. */
  kind?: "graph" | "node";
}

export default function SaveGraphDialog({
  onClose,
  onSaved,
  kind = "graph",
}: SaveGraphDialogProps) {
  // Pre-fill from active tab
  const activeTab = useFlowStore((s) => s.tabs[s.activeTabId]);
  const [name, setName] = useState(activeTab?.title || "");
  const [description, setDescription] = useState(activeTab?.description || "");
  const [group, setGroup] = useState("");
  const [existingGroups, setExistingGroups] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canvasStack = useFlowStore((s) => s.canvasStack);

  // Fetch existing groups from saved graph nodes
  useEffect(() => {
    if (kind !== "node") return;
    api
      .listGraphs()
      .then((all: SavedGraph[]) => {
        const groups = [
          ...new Set(
            all
              .filter((g) => g.kind === "node" && g.group)
              .map((g) => g.group as string),
          ),
        ].sort();
        setExistingGroups(groups);
      })
      .catch(() => {});
  }, [kind]);

  const handleSave = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    setSaving(true);
    setError(null);

    try {
      // Assemble the full graph payload (nodes + edges + containers +
      // access_grants). Subgraph save reads from the stored inner graph
      // snapshot; root save splits the live canvas via flowToGraph and
      // merges in the tab's graph_state slice.
      let nodes: unknown[];
      let edges: unknown[];
      let containers: unknown[];
      let accessGrants: unknown[];

      if (canvasStack.length > 0) {
        const activeId = useFlowStore.getState().activeSubgraphNodeId;
        const activeNode = useFlowStore
          .getState()
          .nodes.find((n) => n.id === activeId);
        const sub = (activeNode?.data?.innerGraph ||
          activeNode?.data?.subgraph) as Record<string, unknown> | undefined;
        nodes = (sub?.nodes as unknown[]) || [];
        edges = (sub?.edges as unknown[]) || [];
        containers = (sub?.containers as unknown[]) || [];
        accessGrants = (sub?.access_grants as unknown[]) || [];
      } else {
        const state = useFlowStore.getState();
        const flat = flowToGraph(state.nodes, state.edges);
        nodes = flat.nodes as unknown[];
        edges = flat.edges as unknown[];
        containers = mergeGraphState(
          flat.containers || [],
          state.tabs[state.activeTabId]?.graphState ?? null,
        ) as unknown[];
        accessGrants = (flat.access_grants || []) as unknown[];
      }

      const result = await api.saveGraph({
        name: name.trim(),
        description: description.trim(),
        nodes,
        edges,
        containers,
        access_grants: accessGrants,
        step_budget: 500,
        kind,
        ...(kind === "node" && group.trim() ? { group: group.trim() } : {}),
      });
      // Update active tab metadata. `description` is captured here so the
      // next in-place Save preserves it without re-prompting.
      const state = useFlowStore.getState();
      const patch: {
        title: string;
        graphId: string;
        dirty: boolean;
        description?: string;
      } = {
        title: name.trim(),
        graphId: result.id,
        dirty: false,
      };
      if (kind === "graph") patch.description = description.trim();
      state.updateTabMeta(state.activeTabId, patch);
      onSaved();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">
            {kind === "node" ? "Save as Graph Node" : "Save Graph"}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300"
          >
            <X size={14} />
          </button>
        </div>

        <div className="mb-3 space-y-2">
          <div>
            <label className="mb-1 block text-xs text-gray-400">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Agent Graph"
              autoFocus
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-gray-400">
              Description
            </label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
          {kind === "node" && (
            <div>
              <label className="mb-1 block text-xs text-gray-400">Group</label>
              <input
                type="text"
                value={group}
                onChange={(e) => setGroup(e.target.value)}
                placeholder="Type new or pick below"
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none"
              />
              {existingGroups.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {existingGroups.map((g) => (
                    <button
                      key={g}
                      type="button"
                      onClick={() => setGroup(g)}
                      className={`rounded-full px-2 py-0.5 text-[10px] transition ${
                        group === g
                          ? "bg-indigo-600 text-white"
                          : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300"
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {kind === "node" && (
          <div className="mb-2 text-[10px] text-gray-500">
            Archived as a reusable node. Drag onto any canvas. Future edits to
            this graph won't affect existing copies.
          </div>
        )}

        {error && <div className="mb-2 text-xs text-red-400">{error}</div>}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !name.trim()}
            className="flex items-center gap-1 rounded bg-blue-600 px-3 py-1.5 text-xs text-white hover:bg-blue-500 disabled:opacity-40"
          >
            <Save size={12} />
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
