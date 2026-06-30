import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { evalApi } from "./evalApi";
import type { GraphIntrospection } from "./types";

interface Props {
  value: string;
  onChange: (
    graphName: string,
    introspection: GraphIntrospection | null,
  ) => void;
  disabled?: boolean;
}

interface GraphEntry {
  _id: string; // file stem (e.g. "navgpt_ce")
  name: string; // display name
  kind?: string; // "graph" (default) or "node" — omitted when "graph"
}

export default function GraphSelector({ value, onChange, disabled }: Props) {
  const [graphs, setGraphs] = useState<GraphEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [introspecting, setIntrospecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    evalApi
      .listGraphs()
      .then((g) =>
        setGraphs((g || []).filter((x) => !x.kind || x.kind === "graph")),
      )
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  async function handleChange(name: string) {
    onChange(name, null);
    if (!name) return;
    setIntrospecting(true);
    try {
      const result = await evalApi.introspectGraph(name);
      onChange(name, result);
    } catch {
      // introspection failed — pass null so parent knows
    } finally {
      setIntrospecting(false);
    }
  }

  const inputCls =
    "mt-1 w-full bg-gray-800 text-gray-200 text-sm px-2 py-1 rounded border border-gray-700";

  return (
    <label className="text-xs text-gray-400">
      Graph
      <div className="relative">
        <select
          value={value}
          onChange={(e) => handleChange(e.target.value)}
          disabled={disabled || loading || introspecting}
          className={inputCls}
        >
          <option value="">— select a graph —</option>
          {graphs.map((g) => (
            <option key={g._id} value={g._id}>
              {g.name}
            </option>
          ))}
        </select>
        {(loading || introspecting) && (
          <Loader2
            size={12}
            className="absolute right-2 top-1/2 -translate-y-1/2 animate-spin text-gray-400"
          />
        )}
      </div>
      {error && <span className="text-xs text-red-400">{error}</span>}
    </label>
  );
}
