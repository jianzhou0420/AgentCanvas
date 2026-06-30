/** Text display field for viewer nodes.
 *
 * Handles both scalar and array inputs:
 *  - scalar TEXT (TextViewerSink)    → single block, latest value
 *  - array of TEXT (TextScrollSink)  → scrollable stack, oldest→newest,
 *    with a "+N earlier" badge when trimmed by max_visible
 */

import type { DisplayFieldProps } from "../viewerFieldRegistry";

function toText(v: unknown): string {
  return typeof v === "string" ? v : v != null ? String(v) : "";
}

export default function TextViewerField({
  value,
  label,
  max_visible,
}: DisplayFieldProps) {
  const isArray = Array.isArray(value);
  const entries = isArray ? (value as unknown[]).map(toText) : [];
  const limit = max_visible && max_visible > 0 ? max_visible : 20;
  const visible = isArray ? entries.slice(-limit) : [];
  const overflow = isArray ? entries.length - visible.length : 0;
  const scalar = isArray ? "" : toText(value);
  const hasContent = isArray ? entries.length > 0 : scalar.length > 0;

  return (
    <div>
      {label && <div className="mb-0.5 text-[9px] text-gray-500">{label}</div>}
      {!hasContent ? (
        <div className="italic text-gray-600 text-[10px]">
          Waiting for data...
        </div>
      ) : isArray ? (
        <div className="max-h-60 space-y-1 overflow-auto rounded bg-gray-800 p-1.5">
          {overflow > 0 && (
            <div className="text-[9px] text-gray-500">+{overflow} earlier</div>
          )}
          {visible.map((text, i) => (
            <div
              key={i}
              className="whitespace-pre-wrap border-t border-gray-700/60 pt-1 text-[10px] text-gray-300 first:border-t-0 first:pt-0"
            >
              {text}
            </div>
          ))}
        </div>
      ) : (
        <div className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-800 p-1.5 text-[10px] text-gray-300">
          {scalar}
        </div>
      )}
    </div>
  );
}
