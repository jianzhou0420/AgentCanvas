/** Tab bar — VS Code-style tabs for multiple graph editing sessions. */

import { Plus, X } from "lucide-react";
import clsx from "clsx";
import { useFlowStore } from "../useFlowStore";

export default function TabBar() {
  const tabs = useFlowStore((s) => s.tabs);
  const activeTabId = useFlowStore((s) => s.activeTabId);
  const tabOrder = useFlowStore((s) => s.tabOrder);
  const setActiveTab = useFlowStore((s) => s.setActiveTab);
  const closeTab = useFlowStore((s) => s.closeTab);
  const newTab = useFlowStore((s) => s.newTab);

  const handleClose = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    const tab = tabs[tabId];
    if (tab?.dirty) {
      if (!window.confirm(`"${tab.title}" has unsaved changes. Close anyway?`))
        return;
    }
    closeTab(tabId);
  };

  return (
    <div className="flex h-8 items-stretch border-b border-gray-800 bg-gray-900">
      {/* Tab strip — scrollable */}
      <div className="flex flex-1 items-stretch overflow-x-auto">
        {tabOrder.map((tabId) => {
          const tab = tabs[tabId];
          if (!tab) return null;
          const isActive = tabId === activeTabId;

          return (
            <button
              key={tabId}
              onClick={() => setActiveTab(tabId)}
              className={clsx(
                "group relative flex items-center gap-1.5 border-r border-gray-800 px-3 text-xs transition-colors",
                "min-w-[100px] max-w-[180px] shrink-0",
                isActive
                  ? "bg-gray-800 text-white"
                  : "bg-gray-900 text-gray-500 hover:bg-gray-800/50 hover:text-gray-300",
              )}
            >
              {/* Active indicator */}
              {isActive && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
              )}

              {/* Dirty indicator */}
              {tab.dirty && (
                <span className="text-[10px] text-yellow-400">●</span>
              )}

              {/* Title */}
              <span className="truncate">{tab.title || "Untitled"}</span>

              {/* Close button */}
              <span
                onClick={(e) => handleClose(e, tabId)}
                className={clsx(
                  "ml-auto flex h-4 w-4 shrink-0 items-center justify-center rounded",
                  "text-gray-600 hover:bg-gray-700 hover:text-gray-300",
                  !isActive && "opacity-0 group-hover:opacity-100",
                )}
              >
                <X size={10} />
              </span>
            </button>
          );
        })}
      </div>

      {/* New tab button */}
      <button
        onClick={() => newTab()}
        className="flex w-8 shrink-0 items-center justify-center border-l border-gray-800 text-gray-600 hover:bg-gray-800 hover:text-gray-300"
        title="New tab"
      >
        <Plus size={14} />
      </button>
    </div>
  );
}
