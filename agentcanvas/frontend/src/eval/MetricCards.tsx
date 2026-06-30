interface Props {
  aggregate_metrics: Record<string, number>;
}

const PRIMARY_METRICS = [
  "spl",
  "success",
  "ndtw",
  "sdtw",
  "path_length",
  "distance_to_goal",
];

export default function MetricCards({ aggregate_metrics }: Props) {
  const metrics = aggregate_metrics ?? {};
  const keys = PRIMARY_METRICS.filter((k) => metrics[k] !== undefined);

  // Also show any keys not in the primary list, up to 6 total
  const extra = Object.keys(metrics)
    .filter((k) => !PRIMARY_METRICS.includes(k))
    .slice(0, Math.max(0, 6 - keys.length));

  const displayKeys = [...keys, ...extra].slice(0, 6);

  if (displayKeys.length === 0) {
    return (
      <div className="py-4 text-center text-xs text-gray-500">
        No metrics yet
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {displayKeys.map((k) => (
        <div
          key={k}
          className="flex flex-col items-center rounded border border-gray-800 bg-gray-800/50 px-3 py-2"
        >
          <span className="text-xs uppercase tracking-wider text-gray-400">
            {k}
          </span>
          <span className="mt-1 font-mono text-xl font-semibold text-gray-100">
            {metrics[k].toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}
