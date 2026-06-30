/** Expand panel for iterIn nodes.
 *
 * Two-sided model: iterIn's left/input side is authored here as
 * ``iterIn.data.initPorts`` (the editable "Init inputs" list). The lower
 * section shows the full synthesised surface (init + iterOut origins) with
 * origin badges and per-port ``persist`` checkboxes; init-origin persist
 * edits route to ``initPorts``, iterOut-origin to the paired iterOut. Any
 * edit triggers a resync so iterIn's handles update live.
 */

import { useCallback } from "react";
import { resolveInstancePorts } from "../../../../portResolution";
import { useFlowStore } from "../../../../useFlowStore";
import {
  PortListEditor,
  type PortEntry as EditablePort,
} from "./PortListEditor";

interface PortEntry {
  name: string;
  wire_type: string;
  persist?: boolean;
  origin?: "init" | "iterOut";
  writer_name?: string;
}

const ORIGIN_LABEL: Record<string, string> = {
  init: "init",
  iterOut: "iter",
};

export function IterInExpandPanel({
  iterInId,
  ports,
}: {
  iterInId: string;
  ports: PortEntry[];
}) {
  const visibleNodes = useFlowStore((s) => s.visibleNodes);
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const resyncIterInPorts = useFlowStore((s) => s.resyncIterInPorts);
  const removeEdgesWhere = useFlowStore((s) => s.removeEdgesWhere);

  // iterIn's own authored init ports (two-sided model). Edited via the
  // "Init inputs" PortListEditor below; synthesised into "init_<name>" handles.
  const iterInNode = visibleNodes.find((n) => n.id === iterInId);
  const initPorts =
    ((iterInNode?.data as Record<string, unknown> | undefined)?.initPorts as
      | EditablePort[]
      | undefined) || [];

  const onInitPortsChange = useCallback(
    (newPorts: EditablePort[]) => {
      // Clean up dangling seed edges into iterIn for any removed init port —
      // the seed wires into the synthesised "init_<name>" handle.
      const oldNames = new Set(initPorts.map((p) => p.name));
      const removed = [...oldNames].filter(
        (n) => !newPorts.some((p) => p.name === n),
      );
      if (removed.length > 0) {
        const removedHandles = new Set(removed.map((n) => `init_${n}`));
        removeEdgesWhere(
          (e) =>
            e.target === iterInId &&
            !!e.targetHandle &&
            removedHandles.has(e.targetHandle),
        );
      }
      updateNodeData(iterInId, { initPorts: newPorts });
      resyncIterInPorts(iterInId);
    },
    [iterInId, initPorts, removeEdgesWhere, updateNodeData, resyncIterInPorts],
  );

  const togglePersist = useCallback(
    (
      writerName: string | undefined,
      origin: "init" | "iterOut" | undefined,
      next: boolean,
    ) => {
      if (!writerName || !origin) return;
      const iterIn = visibleNodes.find((n) => n.id === iterInId);
      if (!iterIn) return;
      const pairedOutId = (iterIn.data as Record<string, unknown>)
        .pairedWith as string | undefined;

      const writers: string[] = [];
      if (origin === "iterOut" && pairedOutId) {
        writers.push(pairedOutId);
      }

      for (const wid of writers) {
        const w = visibleNodes.find((n) => n.id === wid);
        if (!w) continue;
        const wPorts =
          ((w.data as Record<string, unknown>).ports as
            | PortEntry[]
            | undefined) || [];
        const updated = wPorts.map((p) =>
          p.name === writerName ? { ...p, persist: next } : p,
        );
        const schema = (w.data as Record<string, unknown>)._schema as
          | Record<string, unknown>
          | undefined;
        // Rebuild via resolveInstancePorts so iterOut's derived surfaces
        // (stop input, final_* outputs) survive a persist toggle.
        const nextSchema = schema
          ? (resolveInstancePorts(schema, { ports: updated }) as Record<
              string,
              unknown
            >)
          : schema;
        const patch: Record<string, unknown> = { ports: updated };
        if (nextSchema) patch._schema = nextSchema;
        updateNodeData(wid, patch);
      }
      if (origin === "init") {
        // Two-sided model: init ports live on the iterIn itself, so route the
        // persist edit there too (matched by the unprefixed writer name).
        const cur =
          ((iterIn.data as Record<string, unknown>).initPorts as
            | PortEntry[]
            | undefined) || [];
        if (cur.some((p) => p.name === writerName)) {
          updateNodeData(iterInId, {
            initPorts: cur.map((p) =>
              p.name === writerName ? { ...p, persist: next } : p,
            ),
          });
        }
      }
      resyncIterInPorts(iterInId);
    },
    [iterInId, visibleNodes, updateNodeData, resyncIterInPorts],
  );

  return (
    <div className="nopan nodrag space-y-1.5">
      {/* Editable init inputs — iterIn's left/input side run-start slots. */}
      <div className="space-y-0.5">
        <div className="text-[9px] font-semibold uppercase tracking-wider text-gray-500">
          Init inputs (run-start)
        </div>
        <div className="text-[8px] text-gray-600">
          Declare run-start input slots; wire seeds into the matching left-side
          handle. persist=ON keeps the value across iterations.
        </div>
        <PortListEditor
          ports={initPorts}
          onChange={onInitPortsChange}
          showPersist
          defaultPersist={false}
        />
      </div>

      {ports.length === 0 ? (
        <div className="text-[9px] text-gray-500">
          No synthesised ports yet — add an init input above or declare ports on
          the paired iterOut node.
        </div>
      ) : (
        <div className="space-y-0.5">
          <div className="text-[9px] text-gray-500">
            Full loop-carry surface (init + iterOut). Toggle persist to control
            whether a slot retains its value across iterations.
          </div>
          <div className="flex items-center gap-1 text-[8px] uppercase tracking-wider text-gray-600">
            <div className="w-5 shrink-0">src</div>
            <div className="flex-1">name</div>
            <div className="w-12 shrink-0">type</div>
            <div className="w-12 shrink-0 text-right">persist</div>
          </div>
          {ports.map((p) => (
            <div
              key={p.name}
              className="flex items-center gap-1 rounded bg-gray-800/40 px-1 py-0.5"
            >
              <div
                className="w-8 shrink-0 rounded bg-gray-900 text-center text-[8px] text-gray-400"
                title={`origin: ${p.origin ?? "?"}`}
              >
                {ORIGIN_LABEL[p.origin ?? ""] ?? "?"}
              </div>
              <div
                className="flex-1 truncate text-[9px] text-gray-200"
                title={p.writer_name ? `writer port: ${p.writer_name}` : p.name}
              >
                {p.name}
              </div>
              <div className="w-12 shrink-0 truncate text-[8px] text-gray-500">
                {p.wire_type}
              </div>
              <div className="w-12 shrink-0 text-right">
                <input
                  type="checkbox"
                  checked={p.persist !== false}
                  onChange={(e) =>
                    togglePersist(p.writer_name, p.origin, e.target.checked)
                  }
                  className="h-3 w-3"
                  disabled={!p.writer_name || !p.origin}
                  title={
                    p.origin === "init"
                      ? "Writes persist flag to the init port"
                      : p.origin === "iterOut"
                        ? "Writes persist flag to iterOut"
                        : "Direct canvas edge — no writer to toggle"
                  }
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
