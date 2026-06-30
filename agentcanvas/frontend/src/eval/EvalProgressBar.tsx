import clsx from "clsx";
import type { EvalStatus } from "./types";

interface Props {
  completed_count: number;
  total_episodes: number;
  status: EvalStatus;
}

export default function EvalProgressBar({
  completed_count,
  total_episodes,
  status,
}: Props) {
  if (status === "none") return null;

  const pct = total_episodes > 0 ? (completed_count / total_episodes) * 100 : 0;

  const barColor = clsx(
    "h-2.5 rounded transition-all duration-300",
    status === "completed" && "bg-green-500",
    status === "cancelled" && "bg-orange-500",
    (status === "running" || status === "pending") && "bg-blue-500",
    status === "error" && "bg-red-500",
  );

  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-xs text-gray-400">
        <span>
          {completed_count}/{total_episodes > 0 ? total_episodes : "?"} episodes
        </span>
        <span>{total_episodes > 0 ? `${pct.toFixed(0)}%` : ""}</span>
      </div>
      <div className="h-2.5 w-full rounded bg-gray-800">
        <div className={barColor} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
    </div>
  );
}
