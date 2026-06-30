/** Image display field for viewer nodes.
 * Renders a base64-encoded PNG image. */

import type { DisplayFieldProps } from "../viewerFieldRegistry";

export default function ImageViewerField({ value, label }: DisplayFieldProps) {
  const src = typeof value === "string" ? value : "";
  return (
    <div className="overflow-hidden rounded border border-gray-700 bg-gray-800">
      <div className="border-b border-gray-700 px-1.5 py-0.5 text-center text-[10px] text-gray-500">
        {label}
      </div>
      {src ? (
        <img
          src={`data:image/png;base64,${src}`}
          alt={label}
          className="h-24 w-full object-contain"
        />
      ) : (
        <div className="flex h-24 items-center justify-center text-xs text-gray-600">
          No image
        </div>
      )}
    </div>
  );
}
