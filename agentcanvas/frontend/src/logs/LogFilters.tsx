/** Filter toolbar for log entries. */

import clsx from "clsx";
import type { LogViewMode } from "./LogContext";

export interface LogFilterState {
  search: string;
  nodeType: string;
  errorsOnly: boolean;
}

interface Props {
  filters: LogFilterState;
  onChange: (filters: LogFilterState) => void;
  nodeTypes: string[];
  totalCount: number;
  filteredCount: number;
  viewMode: LogViewMode;
  onViewModeChange: (mode: LogViewMode) => void;
}

export default function LogFilters({
  filters,
  onChange,
  nodeTypes,
  totalCount,
  filteredCount,
  viewMode,
  onViewModeChange,
}: Props) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800">
      <input
        type="text"
        placeholder="Filter nodes..."
        value={filters.search}
        onChange={(e) => onChange({ ...filters, search: e.target.value })}
        className="flex-1 bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700 focus:border-blue-500 outline-none min-w-0"
      />

      <select
        value={filters.nodeType}
        onChange={(e) => onChange({ ...filters, nodeType: e.target.value })}
        className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700"
      >
        <option value="">All types</option>
        {nodeTypes.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>

      <button
        onClick={() =>
          onChange({ ...filters, errorsOnly: !filters.errorsOnly })
        }
        className={clsx(
          "px-2 py-1 text-xs rounded border",
          filters.errorsOnly
            ? "border-red-600 bg-red-600/20 text-red-300"
            : "border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200",
        )}
      >
        Errors
      </button>

      {(filters.search || filters.nodeType || filters.errorsOnly) && (
        <button
          onClick={() =>
            onChange({ search: "", nodeType: "", errorsOnly: false })
          }
          className="text-gray-500 hover:text-gray-300 text-xs px-1"
        >
          Clear
        </button>
      )}

      <div className="ml-auto flex items-center gap-0.5 shrink-0">
        <button
          onClick={() => onViewModeChange("overall")}
          className={clsx(
            "px-2 py-1 text-xs rounded-l border",
            viewMode === "overall"
              ? "border-blue-600 bg-blue-600/20 text-blue-300"
              : "border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200",
          )}
        >
          Overall
        </button>
        <button
          onClick={() => onViewModeChange("detail")}
          className={clsx(
            "px-2 py-1 text-xs border border-l-0",
            viewMode === "detail"
              ? "border-blue-600 bg-blue-600/20 text-blue-300"
              : "border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200",
          )}
        >
          Detail
        </button>
        <button
          onClick={() => onViewModeChange("canvas")}
          className={clsx(
            "px-2 py-1 text-xs rounded-r border border-l-0",
            viewMode === "canvas"
              ? "border-blue-600 bg-blue-600/20 text-blue-300"
              : "border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200",
          )}
        >
          Canvas
        </button>
      </div>

      <span className="text-gray-600 text-[10px] shrink-0">
        {filteredCount === totalCount
          ? `${totalCount}`
          : `${filteredCount}/${totalCount}`}{" "}
        entries
      </span>
    </div>
  );
}
