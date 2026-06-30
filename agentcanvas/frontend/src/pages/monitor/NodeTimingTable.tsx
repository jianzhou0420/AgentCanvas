/**
 * NodeTimingTable — unified per-run timing breakdown for a single run.
 *
 * One table, two sections compared on the SAME basis (seconds + share):
 *   • node compute rows (top), then a separator,
 *   • server-mode transport rows (below).
 * Each row shows avg / p5 / p95 (per-firing spread) + total (sum) + share.
 * Share is each row's total as a % of the GRAND total (Σ compute + Σ transport),
 * so the compute and transport shares together sum to 100%.
 *
 * Data from GET /api/system/runs/{id}.node_timing (compute_ms / transport_ms
 * recorded by GraphExecutor — same for canvas Play and every eval episode).
 */

interface NodeTiming {
  node_type: string;
  count: number;
  compute_ms: { mean: number; p5: number; p95: number; total: number };
  transport_ms?: {
    total: number;
    mean: number;
    p5: number;
    p95: number;
  } | null;
}

interface URow {
  key: string;
  type: string;
  count: number;
  avgMs: number;
  p5Ms: number;
  p95Ms: number;
  totalMs: number;
  share: number;
}

function fmtS(ms: number): string {
  const s = ms / 1000;
  if (s >= 10) return s.toFixed(1);
  if (s >= 1) return s.toFixed(2);
  return s.toFixed(3);
}

function fmtPct(p: number): string {
  if (p < 0.05) return "0%";
  return p >= 10 ? `${p.toFixed(0)}%` : `${p.toFixed(1)}%`;
}

function timingRow(r: URow) {
  return (
    <tr key={r.key} className="border-b border-gray-800 last:border-b-0">
      <td className="px-3 py-1.5 font-mono text-gray-200">{r.type}</td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-400">
        {r.count}
      </td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-300">
        {fmtS(r.avgMs)}
      </td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-500">
        {fmtS(r.p5Ms)}
      </td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-300">
        {fmtS(r.p95Ms)}
      </td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-300">
        {fmtS(r.totalMs)}
      </td>
      <td className="px-3 py-1.5 text-right tabular-nums text-gray-400">
        <div className="flex items-center justify-end gap-2">
          <div className="h-1.5 w-16 rounded bg-gray-800">
            <div
              className="h-full rounded bg-sky-500"
              style={{ width: `${Math.min(100, r.share)}%` }}
            />
          </div>
          <span className="w-10">{fmtPct(r.share)}</span>
        </div>
      </td>
    </tr>
  );
}

export default function NodeTimingTable({ rows }: { rows: NodeTiming[] }) {
  if (!rows.length) {
    return (
      <div className="rounded border border-gray-800 bg-gray-900 px-3 py-3 text-xs text-gray-500">
        No node timing recorded for this run.
      </div>
    );
  }

  // Grand total = compute + transport → shares across both sections sum to 100%.
  const grand =
    rows.reduce((a, r) => a + r.compute_ms.total, 0) +
    rows.reduce((a, r) => a + (r.transport_ms?.total ?? 0), 0);
  const pct = (ms: number) => (grand > 0 ? (ms / grand) * 100 : 0);

  const computeRows: URow[] = [...rows]
    .sort((a, b) => b.compute_ms.total - a.compute_ms.total)
    .map((r) => ({
      key: `c:${r.node_type}`,
      type: r.node_type,
      count: r.count,
      avgMs: r.compute_ms.mean,
      p5Ms: r.compute_ms.p5,
      p95Ms: r.compute_ms.p95,
      totalMs: r.compute_ms.total,
      share: pct(r.compute_ms.total),
    }));

  const transportRows: URow[] = rows
    .filter((r) => r.transport_ms)
    .sort((a, b) => b.transport_ms!.total - a.transport_ms!.total)
    .map((r) => ({
      key: `t:${r.node_type}`,
      type: r.node_type,
      count: r.count,
      avgMs: r.transport_ms!.mean,
      p5Ms: r.transport_ms!.p5,
      p95Ms: r.transport_ms!.p95,
      totalMs: r.transport_ms!.total,
      share: pct(r.transport_ms!.total),
    }));

  return (
    <div className="overflow-x-auto rounded border border-gray-800 bg-gray-900">
      <table className="w-full text-left text-xs">
        <thead className="text-gray-500">
          <tr className="border-b border-gray-800">
            <th className="px-3 py-2 font-medium">Type</th>
            <th className="px-3 py-2 text-right font-medium">Count</th>
            <th className="px-3 py-2 text-right font-medium">avg (s)</th>
            <th className="px-3 py-2 text-right font-medium">p5 (s)</th>
            <th className="px-3 py-2 text-right font-medium">p95 (s)</th>
            <th className="px-3 py-2 text-right font-medium">total (s)</th>
            <th className="px-3 py-2 text-right font-medium">share</th>
          </tr>
        </thead>
        <tbody>
          {computeRows.map(timingRow)}
          {/* separator: transport compared below, same basis (seconds + share) */}
          <tr className="bg-gray-950/60">
            <td
              colSpan={7}
              className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-gray-500"
            >
              Transport (server-mode)
            </td>
          </tr>
          {transportRows.length ? (
            transportRows.map(timingRow)
          ) : (
            <tr>
              <td colSpan={7} className="px-3 py-1.5 text-[11px] text-gray-600">
                none — all nodes ran in-process
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <div className="px-3 py-1.5 text-[10px] text-gray-600">
        compute = node forward() · transport = server round-trip · seconds ·
        p5/p95 = per-firing spread · share = % of total (compute + transport)
      </div>
    </div>
  );
}
