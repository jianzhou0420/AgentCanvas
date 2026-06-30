/** State container canvas node — visible shared state on the canvas.
 *
 * Renders a dict of named states (accumulator, lastWrite, counter, ephemeral)
 * with live value previews updated via WebSocket. Nodes connect to containers
 * via state edges (a separate wire system) for read/write access.
 */

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { Database } from "lucide-react";
import clsx from "clsx";

const STATE_TYPE_COLORS: Record<string, string> = {
  accumulator: "bg-amber-500/20 text-amber-400 border-amber-500/50",
  lastWrite: "bg-blue-500/20 text-blue-400 border-blue-500/50",
  counter: "bg-green-500/20 text-green-400 border-green-500/50",
  ephemeral: "bg-gray-500/20 text-gray-400 border-gray-500/50",
};

const STATE_TYPE_BADGE_COLORS: Record<string, string> = {
  accumulator: "bg-amber-600/40 text-amber-300",
  lastWrite: "bg-blue-600/40 text-blue-300",
  counter: "bg-green-600/40 text-green-300",
  ephemeral: "bg-gray-600/40 text-gray-300",
};

// Lifetime declares *when* a state clears (orthogonal to the reducer).
// Forever = no badge to avoid clutter on the common case.
const LIFETIME_BADGE_COLORS: Record<string, string> = {
  step: "bg-gray-700/50 text-gray-300",
  episode: "bg-emerald-700/50 text-emerald-300",
  run: "bg-cyan-700/50 text-cyan-300",
  custom: "bg-pink-700/50 text-pink-300",
};

interface StateEntry {
  type: string;
  value_type: string;
  config?: Record<string, unknown>;
  lifetime?: "step" | "episode" | "run" | "forever" | "custom";
  reset_on?: string[];
}

interface ContainerData {
  label?: string;
  states?: Record<string, StateEntry>;
  // Live preview data (from WS)
  _preview?: Record<
    string,
    {
      type: string;
      value_type: string;
      size?: number;
      value?: number;
      preview?: string;
    }
  >;
  [key: string]: unknown;
}

function StateContainerNode({ data, selected }: NodeProps) {
  const d = data as ContainerData;
  const label = d.label || "State Container";
  const states = d.states || {};
  const preview = d._preview || {};

  return (
    <div
      className={clsx(
        "min-w-[180px] rounded-lg border-2 border-dashed px-3 py-2",
        "bg-gray-900/90 backdrop-blur-sm",
        selected
          ? "border-violet-400 shadow-lg shadow-violet-500/20"
          : "border-violet-500/40",
      )}
    >
      {/* State edge handles — top/bottom for vertical connections */}
      <Handle
        type="target"
        position={Position.Top}
        id="state"
        className="!h-3 !w-3 !rounded-full !border-2 !border-violet-400 !bg-violet-600"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="state"
        className="!h-3 !w-3 !rounded-full !border-2 !border-violet-400 !bg-violet-600"
      />

      {/* Header */}
      <div className="mb-1.5 flex items-center gap-1.5">
        <Database size={14} className="text-violet-400" />
        <span className="text-xs font-semibold text-violet-300">{label}</span>
      </div>

      {/* State entries */}
      {Object.keys(states).length > 0 ? (
        <div className="space-y-1">
          {Object.entries(states).map(([name, entry]) => {
            const liveData = preview[name];
            const typeColor =
              STATE_TYPE_BADGE_COLORS[entry.type] ||
              STATE_TYPE_BADGE_COLORS.lastWrite;
            return (
              <div
                key={name}
                className="flex items-start gap-1.5 rounded bg-gray-800/60 px-1.5 py-1"
              >
                <span className="mt-0.5 text-[8px] text-violet-400">●</span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1">
                    <span className="truncate text-[10px] font-medium text-gray-300">
                      {name}
                    </span>
                    <span
                      className={clsx("rounded px-1 text-[8px]", typeColor)}
                    >
                      {entry.type}
                    </span>
                    {entry.lifetime && entry.lifetime !== "forever" && (
                      <span
                        className={clsx(
                          "rounded px-1 text-[8px]",
                          LIFETIME_BADGE_COLORS[entry.lifetime] ||
                            LIFETIME_BADGE_COLORS.custom,
                        )}
                        title={`lifetime: ${entry.lifetime}`}
                      >
                        {entry.lifetime}
                      </span>
                    )}
                    <span className="text-[8px] text-gray-600">
                      {entry.value_type}
                    </span>
                  </div>
                  {/* Live value preview */}
                  {liveData && (
                    <div className="mt-0.5 truncate text-[9px] text-gray-500">
                      {liveData.size !== undefined && `${liveData.size} items`}
                      {liveData.value !== undefined && `${liveData.value}`}
                      {liveData.preview &&
                        !liveData.size &&
                        liveData.value === undefined &&
                        liveData.preview}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-[10px] italic text-gray-600">
          No states defined
        </div>
      )}
    </div>
  );
}

export default memo(StateContainerNode);
