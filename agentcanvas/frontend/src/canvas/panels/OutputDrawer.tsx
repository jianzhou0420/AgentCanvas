/** Bottom panel: tabbed view with State, Logs, Source, and Report.
 * (Properties moved to the right-side panel — see ResizableRightPanel.) */

import { lazy, Suspense, useEffect, useState } from "react";
import clsx from "clsx";
import LogPanel from "./LogPanel";
import ReportPanel from "./ReportPanel";
import StatePanel from "./StatePanel";
import { useErrorStore } from "../../errorStore";

// Lazy: pulls in CodeMirror — only fetched when the Source tab is opened.
const SourcePanel = lazy(() => import("./SourcePanel"));

const TABS = ["State", "Logs", "Source", "Report"] as const;
type Tab = (typeof TABS)[number];

export default function OutputDrawer() {
  const [activeTab, setActiveTab] = useState<Tab>("State");

  // Listen for "show me the Report tab" requests from elsewhere (toolbar bell).
  const focusReportTick = useErrorStore((s) => s.focusReportTick);
  useEffect(() => {
    if (focusReportTick > 0) setActiveTab("Report");
  }, [focusReportTick]);

  // Unread badge for the Report tab title.
  const unread = useErrorStore((s) => s.unreadCount);

  return (
    <div className="flex h-full flex-col bg-gray-900">
      {/* Tab bar */}
      <div className="flex border-b border-gray-800">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => {
              setActiveTab(tab);
              if (tab === "Report") useErrorStore.getState().markAllRead();
            }}
            className={clsx(
              "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition",
              activeTab === tab
                ? "border-b-2 border-blue-500 text-blue-400"
                : "text-gray-500 hover:text-gray-300",
            )}
          >
            {tab}
            {tab === "Report" && unread > 0 && (
              <span className="rounded-full bg-red-500/80 px-1.5 py-0 text-[9px] font-bold text-white">
                {unread > 99 ? "99+" : unread}
              </span>
            )}
          </button>
        ))}
      </div>
      {/* Content */}
      <div className="min-h-0 flex-1">
        {activeTab === "State" && <StatePanel />}
        {activeTab === "Logs" && <LogPanel />}
        {activeTab === "Source" && (
          <Suspense fallback={null}>
            <SourcePanel />
          </Suspense>
        )}
        {activeTab === "Report" && <ReportPanel />}
      </div>
    </div>
  );
}
