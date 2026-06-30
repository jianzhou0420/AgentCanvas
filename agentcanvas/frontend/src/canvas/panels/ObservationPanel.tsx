/** Observation panel showing RGB + Depth images, extracted from NavigatePage. */

import { useStore } from "../../store";

export default function ObservationPanel() {
  const currentStep = useStore((s) => s.navCurrentStep);
  const rgbBase64 = useStore((s) => s.navCurrentRgb);
  const depthBase64 = useStore((s) => s.navCurrentDepth);

  return (
    <div className="grid h-full grid-cols-2 gap-1">
      <div className="flex flex-col overflow-hidden rounded border border-gray-700 bg-gray-800">
        <div className="flex justify-between border-b border-gray-700 px-2 py-0.5 text-[10px] text-gray-500">
          <span>RGB Camera</span>
          <span className="font-mono">Step {currentStep}</span>
        </div>
        <div className="flex flex-1 items-center justify-center overflow-hidden">
          {rgbBase64 ? (
            <img
              src={`data:image/png;base64,${rgbBase64}`}
              alt="RGB"
              className="max-h-full max-w-full object-contain"
            />
          ) : (
            <div className="text-xs text-gray-600">No observation</div>
          )}
        </div>
      </div>
      <div className="flex flex-col overflow-hidden rounded border border-gray-700 bg-gray-800">
        <div className="flex justify-between border-b border-gray-700 px-2 py-0.5 text-[10px] text-gray-500">
          <span>Depth Map</span>
          <span className="font-mono">Step {currentStep}</span>
        </div>
        <div className="flex flex-1 items-center justify-center overflow-hidden">
          {depthBase64 ? (
            <img
              src={`data:image/png;base64,${depthBase64}`}
              alt="Depth"
              className="max-h-full max-w-full object-contain"
            />
          ) : (
            <div className="text-xs text-gray-600">No observation</div>
          )}
        </div>
      </div>
    </div>
  );
}
