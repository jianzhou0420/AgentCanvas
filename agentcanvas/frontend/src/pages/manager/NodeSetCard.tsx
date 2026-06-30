/** Reusable card for NodeSets and Environments on the Manager page.
 *
 * variant="resource" — compact card for the Resources column (Load button or Active badge)
 * variant="active"   — detailed card for the Active column (Unload button + tool tags)
 * no variant         — original toggle behavior (backward compat)
 */

import { useState } from "react";
import { Package, Power, PowerOff, Loader2, Check } from "lucide-react";
import clsx from "clsx";

interface NodeSetCardProps {
  name: string;
  description: string;
  loaded: boolean;
  tools: string[];
  variant?: "resource" | "active";
  onLoad: () => Promise<void>;
  onUnload: () => Promise<void>;
}

export default function NodeSetCard({
  name,
  description,
  loaded,
  tools,
  variant,
  onLoad,
  onUnload,
}: NodeSetCardProps) {
  const [busy, setBusy] = useState(false);

  const handleAction = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  };

  const isResource = variant === "resource";
  const isActive = variant === "active";

  // Border style depends on variant
  const borderClass = isResource
    ? loaded
      ? "border-green-900/50 bg-gray-800/20"
      : "border-gray-700 bg-gray-800/30"
    : isActive
      ? "border-green-800 bg-gray-800/60"
      : loaded
        ? "border-green-800 bg-gray-800/60"
        : "border-gray-700 bg-gray-800/30";

  return (
    <div className={clsx("rounded-lg border p-4", borderClass)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Package
            size={18}
            className={loaded ? "text-green-400" : "text-gray-500"}
          />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-200">{name}</span>
              <div
                className={clsx(
                  "h-2 w-2 rounded-full",
                  loaded ? "bg-green-500" : "bg-gray-600",
                )}
              />
              <span className="text-xs text-gray-500">
                {loaded ? `Loaded (${tools.length} tools)` : "Not loaded"}
              </span>
            </div>
            <div className="text-sm text-gray-400">{description}</div>
          </div>
        </div>

        {/* Action area — varies by variant */}
        {isResource ? (
          loaded ? (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <Check size={12} /> Active
            </span>
          ) : (
            <button
              onClick={() => handleAction(onLoad)}
              disabled={busy}
              className="flex items-center gap-1.5 rounded bg-green-900/50 px-3 py-1.5 text-sm font-medium text-green-300 hover:bg-green-800/70 disabled:opacity-50"
            >
              {busy ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Power size={14} />
              )}
              Load
            </button>
          )
        ) : isActive ? (
          <button
            onClick={() => handleAction(onUnload)}
            disabled={busy}
            className="flex items-center gap-1.5 rounded bg-gray-700 px-3 py-1.5 text-sm font-medium text-gray-300 hover:bg-red-900/50 hover:text-red-300 disabled:opacity-50"
          >
            {busy ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <PowerOff size={14} />
            )}
            Unload
          </button>
        ) : (
          /* Default: toggle behavior */
          <button
            onClick={() => handleAction(loaded ? onUnload : onLoad)}
            disabled={busy}
            className={clsx(
              "flex items-center gap-1.5 rounded px-3 py-1.5 text-sm font-medium transition-colors",
              loaded
                ? "bg-gray-700 text-gray-300 hover:bg-red-900/50 hover:text-red-300"
                : "bg-green-900/50 text-green-300 hover:bg-green-800/70",
              busy && "cursor-not-allowed opacity-50",
            )}
          >
            {busy ? (
              <Loader2 size={14} className="animate-spin" />
            ) : loaded ? (
              <PowerOff size={14} />
            ) : (
              <Power size={14} />
            )}
            {loaded ? "Unload" : "Load"}
          </button>
        )}
      </div>

      {/* Tool tags — hidden in resource variant */}
      {!isResource && loaded && tools.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {tools.map((t) => (
            <span
              key={t}
              className="rounded bg-gray-700/80 px-2 py-0.5 text-xs text-gray-300"
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
