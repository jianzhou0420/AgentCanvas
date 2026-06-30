import { useState } from "react";
import clsx from "clsx";
import type { EvalEpisodeResult } from "./types";

interface Props {
  episodes: EvalEpisodeResult[];
}

type SortKey =
  | "episode_index"
  | "episode_id"
  | "scene_id"
  | "spl"
  | "success"
  | "ndtw"
  | "status"
  | "worker_id";

export default function EpisodesTable({ episodes }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("episode_index");
  const [sortAsc, setSortAsc] = useState(true);

  if (episodes.length === 0) {
    return (
      <div className="py-4 text-center text-xs text-gray-500">
        No episodes yet
      </div>
    );
  }

  // ADR-028 PB-3: only show the Worker column when the run actually used
  // more than one worker. Keeps the single-worker table identical to today.
  const workerIds = new Set(
    episodes
      .map((ep) => ep.worker_id)
      .filter((w): w is number => w !== undefined),
  );
  const showWorker = workerIds.size > 1;

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  const sorted = [...episodes].sort((a, b) => {
    let av: string | number;
    let bv: string | number;
    if (sortKey === "spl") {
      av = a.metrics.spl ?? -1;
      bv = b.metrics.spl ?? -1;
    } else if (sortKey === "success") {
      av = a.metrics.success ?? -1;
      bv = b.metrics.success ?? -1;
    } else if (sortKey === "ndtw") {
      av = a.metrics.ndtw ?? -1;
      bv = b.metrics.ndtw ?? -1;
    } else if (sortKey === "worker_id") {
      av = a.worker_id ?? -1;
      bv = b.worker_id ?? -1;
    } else {
      av = a[sortKey] ?? "";
      bv = b[sortKey] ?? "";
    }
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });

  function SortHeader({ label, colKey }: { label: string; colKey: SortKey }) {
    const active = sortKey === colKey;
    return (
      <th
        className="cursor-pointer select-none px-2 py-1 text-left hover:text-gray-200"
        onClick={() => handleSort(colKey)}
      >
        <span className={clsx(active ? "text-gray-200" : "text-gray-400")}>
          {label}
          {active ? (sortAsc ? " ▲" : " ▼") : ""}
        </span>
      </th>
    );
  }

  function SortHeaderRight({
    label,
    colKey,
  }: {
    label: string;
    colKey: SortKey;
  }) {
    const active = sortKey === colKey;
    return (
      <th
        className="cursor-pointer select-none px-2 py-1 text-right hover:text-gray-200"
        onClick={() => handleSort(colKey)}
      >
        <span className={clsx(active ? "text-gray-200" : "text-gray-400")}>
          {active ? (sortAsc ? "▲ " : "▼ ") : ""}
          {label}
        </span>
      </th>
    );
  }

  return (
    <div className="max-h-[400px] overflow-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-900">
          <tr className="border-b border-gray-700">
            <SortHeader label="#" colKey="episode_index" />
            {showWorker && <SortHeader label="Worker" colKey="worker_id" />}
            <SortHeader label="Episode ID" colKey="episode_id" />
            <SortHeader label="Scene" colKey="scene_id" />
            <SortHeaderRight label="SPL" colKey="spl" />
            <SortHeaderRight label="SR" colKey="success" />
            <SortHeaderRight label="nDTW" colKey="ndtw" />
            <SortHeader label="Status" colKey="status" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((ep) => (
            <tr
              key={ep.episode_index}
              className="border-b border-gray-800 hover:bg-gray-800/50"
            >
              <td className="px-2 py-1 text-gray-400">{ep.episode_index}</td>
              {showWorker && (
                <td className="px-2 py-1 font-mono text-gray-400">
                  {ep.worker_id ?? "—"}
                </td>
              )}
              <td className="px-2 py-1 font-mono text-gray-300">
                {ep.episode_id}
              </td>
              <td className="max-w-[120px] truncate px-2 py-1 text-gray-400">
                {ep.scene_id.split("/").pop()}
              </td>
              <td className="px-2 py-1 text-right font-mono text-gray-200">
                {ep.metrics.spl !== undefined
                  ? ep.metrics.spl.toFixed(3)
                  : "---"}
              </td>
              <td className="px-2 py-1 text-right font-mono text-gray-200">
                {ep.metrics.success !== undefined
                  ? ep.metrics.success.toFixed(0)
                  : "---"}
              </td>
              <td className="px-2 py-1 text-right font-mono text-gray-200">
                {ep.metrics.ndtw !== undefined
                  ? ep.metrics.ndtw.toFixed(3)
                  : "---"}
              </td>
              <td className="px-2 py-1">
                <span
                  className={clsx(
                    "text-xs",
                    ep.status === "completed" && "text-green-400",
                    ep.status === "running" && "text-yellow-400",
                    ep.status === "error" && "text-red-400",
                    ep.status === "pending" && "text-gray-500",
                  )}
                >
                  {ep.status}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
