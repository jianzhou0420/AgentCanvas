/** Log Viewer Page — dedicated full-page log inspection. */

import { useState, useCallback, lazy, Suspense } from "react";
import ExecutionList from "./ExecutionList";
import LogSummaryBar from "./LogSummaryBar";
import LogFilters from "./LogFilters";
import type { LogFilterState } from "./LogFilters";
import LogEntryList from "./LogEntryList";
import type { LogViewMode } from "./LogContext";

const LogCanvasView = lazy(() => import("./LogCanvasView"));

export default function LogViewerPage() {
  const [selectedExId, setSelectedExId] = useState<string | null>(null);
  const [filters, setFilters] = useState<LogFilterState>({
    search: "",
    nodeType: "",
    errorsOnly: false,
  });
  const [viewMode, setViewMode] = useState<LogViewMode>("overall");
  const [nodeTypes, setNodeTypes] = useState<string[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [filteredCount, setFilteredCount] = useState(0);

  // TODO: detect active execution from WS events
  const activeExecutionId: string | null = null;

  const handleNodeTypes = useCallback((types: string[]) => {
    setNodeTypes(types);
    setTotalCount(0);
  }, []);

  const handleFilteredCount = useCallback(
    (count: number) => {
      setFilteredCount(count);
      if (totalCount === 0) setTotalCount(count);
    },
    [totalCount],
  );

  return (
    <div className="flex h-full bg-gray-900 text-gray-200">
      {/* Left sidebar — execution list */}
      <div className="w-64 border-r border-gray-800 flex flex-col shrink-0">
        <div className="px-3 py-2 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300">Executions</h2>
        </div>
        <ExecutionList
          selectedId={selectedExId}
          onSelect={setSelectedExId}
          activeExecutionId={activeExecutionId}
        />
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0">
        {selectedExId ? (
          <>
            <LogSummaryBar
              executionId={selectedExId}
              isLive={selectedExId === activeExecutionId}
            />
            <LogFilters
              filters={filters}
              onChange={setFilters}
              nodeTypes={nodeTypes}
              totalCount={totalCount}
              filteredCount={filteredCount}
              viewMode={viewMode}
              onViewModeChange={setViewMode}
            />
            {viewMode === "canvas" ? (
              <Suspense
                fallback={
                  <div className="flex-1 flex items-center justify-center text-gray-600 text-xs">
                    Loading canvas...
                  </div>
                }
              >
                <LogCanvasView executionId={selectedExId} />
              </Suspense>
            ) : (
              <LogEntryList
                executionId={selectedExId}
                filters={filters}
                viewMode={viewMode}
                onFilteredCount={handleFilteredCount}
                onNodeTypes={handleNodeTypes}
              />
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
            Select an execution from the sidebar to view logs.
          </div>
        )}
      </div>
    </div>
  );
}
