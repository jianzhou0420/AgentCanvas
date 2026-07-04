import { useState, useEffect } from "react";
import { useStore } from "../store";
import { Navigation, Settings } from "lucide-react";
import clsx from "clsx";
import SettingsModal from "./SettingsModal";

export default function Header() {
  const connected = useStore((s) => s.connected);
  const appMode = useStore((s) => s.appMode);
  const setAppMode = useStore((s) => s.setAppMode);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [execMode, setExecMode] = useState<string>("idle");

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(
          "http://localhost:8000/api/navigate/execution-mode",
        );
        const data = await r.json();
        setExecMode(data.mode);
      } catch {
        /* ignore */
      }
    };
    check();
    const interval = setInterval(check, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <header className="flex items-center gap-4 border-b border-gray-800 bg-gray-900 px-4 py-2">
        <div className="flex items-center gap-2 text-lg font-bold text-blue-400">
          <Navigation size={20} />
          AgentCanvas
        </div>

        <div className="flex items-center overflow-hidden rounded border border-gray-700">
          <button
            onClick={() => setAppMode("nav")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "nav"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Canvas
          </button>
          <button
            onClick={() => setAppMode("manager")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "manager"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Manager
          </button>
          <button
            onClick={() => setAppMode("eval")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "eval"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Evaluate
          </button>
          <button
            onClick={() => setAppMode("logs")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "logs"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Logs
          </button>
          <button
            onClick={() => setAppMode("replay")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "replay"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Replay
          </button>
          <button
            onClick={() => setAppMode("monitor")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "monitor"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Monitor
          </button>
          <button
            onClick={() => setAppMode("coding")}
            className={clsx(
              "px-3 py-1 text-sm font-medium",
              appMode === "coding"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200",
            )}
          >
            Coding Agent
          </button>
        </div>

        <div className="flex-1" />

        <button
          onClick={() => setSettingsOpen(true)}
          className="rounded p-1 text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          title="Settings"
        >
          <Settings size={18} />
        </button>

        <div className="flex items-center gap-2 text-sm text-gray-400">
          <div
            className={`h-2.5 w-2.5 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
          />
          {connected ? "Connected" : "Disconnected"}
        </div>

        {execMode !== "idle" && (
          <span
            className={clsx(
              "rounded px-1.5 py-0.5 text-xs font-medium",
              execMode === "eval"
                ? "bg-purple-600/30 text-purple-300"
                : "bg-blue-600/30 text-blue-300",
            )}
          >
            {execMode === "eval" ? "Eval Running" : "Canvas Running"}
          </span>
        )}
      </header>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </>
  );
}
