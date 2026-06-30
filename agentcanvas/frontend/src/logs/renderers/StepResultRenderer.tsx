/** Renders STEP_RESULT wire type — decomposes into sub-renderers.
 * STEP_RESULT = {observation: OBSERVATION, state: STATE, done: BOOL, metrics?: METRICS}
 */

import { Suspense } from "react";
import { resolveRenderer } from "./registry";
import type { ValueRendererProps } from "./registry";

const SUB_WIRE_TYPES: Record<string, string> = {
  observation: "OBSERVATION",
  state: "STATE",
  done: "BOOL",
  metrics: "METRICS",
};

export default function StepResultRenderer({ value }: ValueRendererProps) {
  if (value === null || value === undefined || typeof value !== "object") {
    return <span className="text-gray-600 text-[10px]">no step result</span>;
  }

  const obj = value as Record<string, unknown>;
  const entries = Object.entries(obj);

  if (entries.length === 0) {
    return <span className="text-gray-600 text-[10px]">empty step result</span>;
  }

  return (
    <div className="space-y-1.5 border-l-2 border-gray-700 pl-2">
      {entries.map(([key, val]) => {
        const wireType = SUB_WIRE_TYPES[key];
        const Renderer = resolveRenderer(val, wireType);
        return (
          <div key={key}>
            <span className="text-[9px] text-gray-500 font-semibold uppercase">
              {key}
            </span>
            <div className="mt-0.5">
              <Suspense
                fallback={
                  <span className="text-gray-600 text-[10px]">loading...</span>
                }
              >
                <Renderer value={val} label={key} />
              </Suspense>
            </div>
          </div>
        );
      })}
    </div>
  );
}
