import { Eye, Layers } from "lucide-react";

export default function ViewPanel() {
  return (
    <div className="flex h-full flex-col rounded border border-gray-800 bg-gray-900">
      <div className="border-b border-gray-800 px-3 py-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Views
        </h3>
      </div>
      <div className="flex min-h-0 flex-1 gap-1 p-2">
        {/* RGB View */}
        <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded border border-gray-700 bg-gray-800">
          <Eye size={24} className="text-blue-400" />
          <span className="text-xs text-gray-500">RGB Camera</span>
          <span className="text-[10px] text-gray-600">640 x 480</span>
        </div>
        {/* Depth View */}
        <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded border border-gray-700 bg-gray-800">
          <Layers size={24} className="text-purple-400" />
          <span className="text-xs text-gray-500">Depth Map</span>
          <span className="text-[10px] text-gray-600">640 x 480</span>
        </div>
      </div>
    </div>
  );
}
