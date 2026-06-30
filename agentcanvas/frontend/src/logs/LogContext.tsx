/** React context providing executionId and view mode to log renderers. */
import { createContext } from "react";

export type LogViewMode = "overall" | "detail" | "canvas";

export const LogContext = createContext<{
  executionId: string | null;
  viewMode: LogViewMode;
}>({ executionId: null, viewMode: "overall" });
