/** Shared wrapper for all custom nodes: dark card with colored header and status dot. */

import type { ReactNode } from "react";
import clsx from "clsx";
import type { NavStatus } from "../../../types";

interface NodeShellProps {
  title: string;
  icon?: ReactNode;
  color?: string; // tailwind bg class for header accent
  status?: NavStatus | "inactive";
  children: ReactNode;
  className?: string;
}

const STATUS_DOT: Record<string, string> = {
  idle: "bg-gray-500",
  loading: "bg-orange-400 animate-pulse",
  running: "bg-green-400 animate-pulse",
  paused: "bg-yellow-400",
  done: "bg-blue-400",
  error: "bg-red-400",
  inactive: "bg-gray-700",
};

export default function NodeShell({
  title,
  icon,
  color = "bg-gray-700",
  status = "inactive",
  children,
  className,
}: NodeShellProps) {
  return (
    <div
      className={clsx(
        "min-w-[220px] max-w-[300px] rounded-lg border border-gray-700 bg-gray-900 shadow-lg",
        className,
      )}
    >
      {/* Header */}
      <div
        className={clsx(
          "flex items-center gap-2 rounded-t-lg px-3 py-1.5",
          color,
        )}
      >
        {icon}
        <span className="flex-1 text-xs font-semibold text-white">{title}</span>
        <span
          className={clsx(
            "h-2 w-2 rounded-full",
            STATUS_DOT[status] || STATUS_DOT.inactive,
          )}
        />
      </div>
      {/* Body — nopan/nodrag so inputs/buttons are interactive */}
      <div className="nopan nodrag nowheel flex flex-col gap-2 p-3 text-xs text-gray-300">
        {children}
      </div>
    </div>
  );
}
