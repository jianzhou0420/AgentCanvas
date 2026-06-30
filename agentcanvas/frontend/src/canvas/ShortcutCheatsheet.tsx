import { useEffect } from "react";

const ROWS: Array<[string, string]> = [
  ["Space", "Play / Pause the current graph"],
  ["F", "Fit view to all nodes"],
  ["?", "Toggle this cheatsheet"],
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function ShortcutCheatsheet({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="min-w-[280px] rounded border border-gray-700 bg-gray-900 p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
          Keyboard Shortcuts
        </div>
        <div className="space-y-1.5">
          {ROWS.map(([key, label]) => (
            <div
              key={key}
              className="flex items-center justify-between gap-6 text-[11px]"
            >
              <kbd className="rounded border border-gray-700 bg-gray-800 px-2 py-0.5 font-mono text-gray-200">
                {key}
              </kbd>
              <span className="text-gray-400">{label}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 text-right">
          <button
            onClick={onClose}
            className="text-[10px] text-gray-500 hover:text-gray-300"
          >
            Close (Esc)
          </button>
        </div>
      </div>
    </div>
  );
}
