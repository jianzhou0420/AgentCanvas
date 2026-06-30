/** Typed Handle component with color coding by edge type. */

import { Handle, type HandleProps } from "@xyflow/react";
import type { CSSProperties } from "react";

const HANDLE_COLORS: Record<string, string> = {
  env: "#22c55e", // green — environment (includes episode/task)
  stream: "#f97316", // orange — output data stream
};

interface HandleDotProps extends Omit<HandleProps, "style"> {
  handleType: string; // env | task | model | tool | stream
  style?: CSSProperties;
}

export default function HandleDot({
  handleType,
  style,
  ...props
}: HandleDotProps) {
  return (
    <Handle
      {...props}
      id={props.id || handleType}
      style={{
        width: 10,
        height: 10,
        background: HANDLE_COLORS[handleType] || "#6b7280",
        border: "2px solid #1f2937",
        ...style,
      }}
    />
  );
}
