/** Server card for server-mode nodesets on the Manager page.
 *
 * variant="resource" — compact card for Resources column (Start button or Active badge)
 * variant="active"   — detailed card for Active column (Stop/Restart + URL/PID/nodes/errors)
 * no variant         — original behavior (backward compat)
 */

import { useState } from "react";
import { Globe, Play, Square, RotateCw, Loader2, Check } from "lucide-react";
import clsx from "clsx";
import type { ServerStatus } from "../../api";

interface ServerCardProps {
  server: ServerStatus;
  variant?: "resource" | "active";
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  onRestart: () => Promise<void>;
}

const STATUS_COLORS: Record<string, string> = {
  connected: "bg-green-500",
  starting: "bg-yellow-500 animate-pulse",
  stopped: "bg-gray-600",
  unreachable: "bg-orange-500",
  error: "bg-red-500",
};

const STATUS_TEXT_COLORS: Record<string, string> = {
  connected: "text-green-400",
  starting: "text-yellow-400",
  stopped: "text-gray-500",
  unreachable: "text-orange-400",
  error: "text-red-400",
};

export default function ServerCard({
  server,
  variant,
  onStart,
  onStop,
  onRestart,
}: ServerCardProps) {
  const [busy, setBusy] = useState(false);

  const wrap = (fn: () => Promise<void>) => async () => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const isUp = server.status === "connected";
  const isStarting = server.status === "starting";
  const isResource = variant === "resource";
  const isActive = variant === "active";

  const borderClass = isResource
    ? isUp || isStarting
      ? "border-green-900/50 bg-gray-800/20"
      : "border-gray-700 bg-gray-800/30"
    : isActive
      ? "border-green-800 bg-gray-800/60"
      : isUp
        ? "border-green-800 bg-gray-800/60"
        : "border-gray-700 bg-gray-800/30";

  return (
    <div className={clsx("rounded-lg border p-4", borderClass)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Globe
            size={18}
            className={isUp ? "text-green-400" : "text-gray-500"}
          />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-200">{server.name}</span>
              <div
                className={clsx(
                  "h-2 w-2 rounded-full",
                  STATUS_COLORS[server.status] || "bg-gray-600",
                )}
              />
              <span
                className={clsx(
                  "text-xs",
                  STATUS_TEXT_COLORS[server.status] || "text-gray-500",
                )}
              >
                {server.status}
              </span>
            </div>
            <div className="text-sm text-gray-400">{server.description}</div>
          </div>
        </div>

        {/* Action area — varies by variant */}
        {isResource ? (
          isUp || isStarting ? (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <Check size={12} /> Active
            </span>
          ) : (
            <button
              onClick={wrap(onStart)}
              disabled={busy}
              className="flex items-center gap-1.5 rounded bg-green-900/50 px-3 py-1.5 text-sm font-medium text-green-300 hover:bg-green-800/70 disabled:opacity-50"
            >
              {busy ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Play size={14} />
              )}
              Start
            </button>
          )
        ) : isActive ? (
          <div className="flex items-center gap-2">
            <button
              onClick={wrap(onStop)}
              disabled={busy}
              className="flex items-center gap-1.5 rounded bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-red-900/50 hover:text-red-300 disabled:opacity-50"
            >
              {busy ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Square size={14} />
              )}
              Stop
            </button>
            <button
              onClick={wrap(onRestart)}
              disabled={busy}
              className="flex items-center gap-1.5 rounded bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-blue-900/50 hover:text-blue-300 disabled:opacity-50"
            >
              <RotateCw size={14} />
            </button>
          </div>
        ) : (
          /* Default: original behavior */
          <div className="flex items-center gap-2">
            {!isUp && !isStarting && (
              <button
                onClick={wrap(onStart)}
                disabled={busy}
                className="flex items-center gap-1.5 rounded bg-green-900/50 px-3 py-1.5 text-sm font-medium text-green-300 hover:bg-green-800/70 disabled:opacity-50"
              >
                {busy ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Play size={14} />
                )}
                Start
              </button>
            )}
            {(isUp || isStarting) && (
              <>
                <button
                  onClick={wrap(onStop)}
                  disabled={busy}
                  className="flex items-center gap-1.5 rounded bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-red-900/50 hover:text-red-300 disabled:opacity-50"
                >
                  {busy ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Square size={14} />
                  )}
                  Stop
                </button>
                <button
                  onClick={wrap(onRestart)}
                  disabled={busy}
                  className="flex items-center gap-1.5 rounded bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-blue-900/50 hover:text-blue-300 disabled:opacity-50"
                >
                  <RotateCw size={14} />
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {/* Details row — hidden in resource variant */}
      {!isResource && (
        <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
          <span>{server.url}</span>
          {server.pid && <span>PID: {server.pid}</span>}
          {server.auto_restart && (
            <span className="text-yellow-600">auto-restart</span>
          )}
        </div>
      )}

      {/* Error — hidden in resource variant */}
      {!isResource && server.error && (
        <div className="mt-2 rounded bg-red-900/30 px-2 py-1 text-xs text-red-400">
          {server.error}
        </div>
      )}

      {/* Node tags — hidden in resource variant */}
      {!isResource && server.nodes && server.nodes.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {server.nodes.map((n) => (
            <span
              key={n}
              className="rounded bg-gray-700/80 px-2 py-0.5 text-xs text-gray-300"
            >
              {n}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
