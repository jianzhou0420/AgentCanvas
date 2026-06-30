/** Step log panel — shows either straightforward action log or LLM thinking steps. */

import { useEffect, useRef } from "react";
import clsx from "clsx";
import { useStore } from "../../store";

const ACTION_NAMES: Record<number, string> = {
  0: "STOP",
  1: "FORWARD",
  2: "TURN_LEFT",
  3: "TURN_RIGHT",
};

export default function StepLogPanel() {
  const llmSteps = useStore((s) => s.navLLMSteps);
  const isLLM = llmSteps.length > 0;

  return isLLM ? <LLMLog /> : <ActionLog />;
}

function ActionLog() {
  const steps = useStore((s) => s.navSteps);
  const currentStep = useStore((s) => s.navCurrentStep);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps.length]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex justify-between border-b border-gray-700 px-2 py-1 text-[10px] text-gray-500">
        <span className="font-semibold uppercase tracking-wider">
          Action Log
        </span>
        <span>{steps.length} steps</span>
      </div>
      <div className="flex-1 space-y-0.5 overflow-auto p-1.5">
        {steps.length === 0 && (
          <div className="py-2 text-center text-xs italic text-gray-600">
            Waiting...
          </div>
        )}
        {steps.map((s) => (
          <div
            key={s.step}
            className={clsx(
              "flex items-center gap-2 rounded px-1.5 py-0.5 text-[11px]",
              s.step === currentStep && "bg-blue-900/30",
            )}
          >
            <span className="w-5 text-right font-mono text-gray-500">
              {s.step}
            </span>
            <span
              className={clsx(
                "rounded px-1 py-0.5 font-mono text-[10px] font-medium",
                s.action === 0 && "bg-red-900/50 text-red-300",
                s.action === 1 && "bg-green-900/50 text-green-300",
                s.action === 2 && "bg-yellow-900/50 text-yellow-300",
                s.action === 3 && "bg-purple-900/50 text-purple-300",
              )}
            >
              {s.action_name || ACTION_NAMES[s.action]}
            </span>
            {s.done && (
              <span className="text-[10px] font-medium text-red-400">DONE</span>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function LLMLog() {
  const steps = useStore((s) => s.navLLMSteps);
  const currentStep = useStore((s) => s.navCurrentStep);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps.length]);

  const borderColor = (type: string) => {
    switch (type) {
      case "reasoning":
        return "border-l-gray-500";
      case "tool_call":
        return "border-l-blue-500";
      case "tool_result":
        return "border-l-green-500";
      case "decision":
        return "border-l-yellow-500";
      default:
        return "border-l-gray-700";
    }
  };

  const typeLabel = (type: string) => {
    switch (type) {
      case "reasoning":
        return { text: "Reasoning", cls: "text-gray-400" };
      case "tool_call":
        return { text: "Tool Call", cls: "text-blue-400" };
      case "tool_result":
        return { text: "Result", cls: "text-green-400" };
      case "decision":
        return { text: "Decision", cls: "text-yellow-400" };
      default:
        return { text: type, cls: "text-gray-400" };
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex justify-between border-b border-gray-700 px-2 py-1 text-[10px] text-gray-500">
        <span className="font-semibold uppercase tracking-wider">
          Agent Thinking
        </span>
        <span>{steps.length} steps</span>
      </div>
      <div className="flex-1 space-y-1 overflow-auto p-1.5">
        {steps.length === 0 && (
          <div className="py-2 text-center text-xs italic text-gray-600">
            Waiting...
          </div>
        )}
        {steps.map((s, i) => {
          const label = typeLabel(s.type);
          return (
            <div
              key={i}
              className={clsx(
                "border-l-2 py-0.5 pl-2",
                borderColor(s.type),
                s.step === currentStep && "rounded-r bg-gray-800/50",
              )}
            >
              <div className="mb-0.5 flex items-center gap-1">
                <span className={clsx("text-[10px] font-medium", label.cls)}>
                  {label.text}
                </span>
                {s.tool && (
                  <span className="font-mono text-[10px] text-blue-300">
                    {s.tool}
                  </span>
                )}
              </div>
              <div className="whitespace-pre-wrap text-[11px] leading-relaxed text-gray-300">
                {s.content}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
