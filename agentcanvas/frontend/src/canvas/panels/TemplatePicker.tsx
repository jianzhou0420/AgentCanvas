/** Template selection dialog. */

import { TEMPLATES } from "../defaultGraph";
import { useFlowStore } from "../useFlowStore";

interface TemplatePickerProps {
  onClose: () => void;
}

export default function TemplatePicker({ onClose }: TemplatePickerProps) {
  const loadTemplate = useFlowStore((s) => s.loadTemplate);

  const handleSelect = (id: string) => {
    loadTemplate(id);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-lg border border-gray-700 bg-gray-900 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-4 text-lg font-semibold text-gray-200">
          Choose a Template
        </h2>
        <div className="space-y-2">
          {TEMPLATES.map((t) => (
            <button
              key={t.id}
              onClick={() => handleSelect(t.id)}
              className="w-full rounded border border-gray-700 bg-gray-800 p-3 text-left transition hover:border-blue-500 hover:bg-gray-800/80"
            >
              <div className="text-sm font-medium text-gray-200">{t.name}</div>
              <div className="text-xs text-gray-500">{t.description}</div>
              <div className="mt-1 text-[10px] text-gray-600">
                {t.nodes.length} nodes
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
