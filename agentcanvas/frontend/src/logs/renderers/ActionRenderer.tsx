/** Renders ACTION wire type as colored badge.
 * VLN-CE actions: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
 */

import clsx from "clsx";
import type { ValueRendererProps } from "./registry";

const ACTION_MAP: Record<number, { label: string; color: string }> = {
  0: { label: "STOP", color: "bg-red-600/30 text-red-300 border-red-600/50" },
  1: {
    label: "MOVE_FORWARD",
    color: "bg-green-600/30 text-green-300 border-green-600/50",
  },
  2: {
    label: "TURN_LEFT",
    color: "bg-blue-600/30 text-blue-300 border-blue-600/50",
  },
  3: {
    label: "TURN_RIGHT",
    color: "bg-orange-600/30 text-orange-300 border-orange-600/50",
  },
};

export default function ActionRenderer({ value }: ValueRendererProps) {
  const actionId = typeof value === "number" ? value : Number(value);
  const action = ACTION_MAP[actionId];

  if (!action) {
    return (
      <span className="inline-flex items-center rounded border border-gray-600 bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-400">
        action={String(value)}
      </span>
    );
  }

  return (
    <span
      className={clsx(
        "inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium",
        action.color,
      )}
    >
      {action.label}
    </span>
  );
}
