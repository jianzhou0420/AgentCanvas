/** Zustand store + WebSocket subscriptions. */

import { create } from "zustand";
import { api } from "./api";
import { wsManager } from "./ws";
import { getFlowStoreBridge } from "./canvas/flowStoreRef";
import { useErrorStore } from "./errorStore";
import type { ErrorEnvelope } from "./errors";
import type {
  EvalRunSummary,
  EvalEpisodeResult,
  NavStatus,
  NavStepData,
  NavLLMStepData,
} from "./types";

interface AppStore {
  // Connection
  connected: boolean;

  // Eval
  appMode: "nav" | "manager" | "eval" | "logs" | "replay" | "monitor";
  evalRun: EvalRunSummary | null;
  evalEpisodes: EvalEpisodeResult[];

  // Navigate
  navStatus: NavStatus;
  navCurrentStep: number;
  navSteps: NavStepData[];
  navLLMSteps: NavLLMStepData[];
  navCurrentRgb: string;
  navCurrentDepth: string;
  navMetrics: Record<string, number> | null;
  navStepDelay: number;

  // Actions
  setConnected: (v: boolean) => void;

  // Eval actions
  setAppMode: (
    mode: "nav" | "manager" | "eval" | "logs" | "replay" | "monitor",
  ) => void;
  setEvalRun: (run: EvalRunSummary | null) => void;
  addOrUpdateEvalEpisode: (ep: EvalEpisodeResult) => void;
  loadEvalStatus: () => Promise<void>;

  // Navigate actions
  setNavStepDelay: (ms: number) => void;
  navPause: () => Promise<void>;
  navStop: () => Promise<void>;
}

export const useStore = create<AppStore>((set, get) => ({
  connected: false,

  // Eval
  appMode: "nav",
  evalRun: null,
  evalEpisodes: [],

  // Navigate
  navStatus: "idle",
  navCurrentStep: 0,
  navSteps: [],
  navLLMSteps: [],
  navCurrentRgb: "",
  navCurrentDepth: "",
  navMetrics: null,
  navStepDelay: 200,

  setConnected: (v) => set({ connected: v }),

  // Eval actions
  setAppMode: (mode) => set({ appMode: mode }),

  setEvalRun: (run) => set({ evalRun: run }),

  addOrUpdateEvalEpisode: (ep) =>
    set((s) => {
      const existing = s.evalEpisodes.findIndex(
        (e) => e.episode_index === ep.episode_index,
      );
      if (existing >= 0) {
        const updated = [...s.evalEpisodes];
        updated[existing] = ep;
        return { evalEpisodes: updated };
      }
      return { evalEpisodes: [...s.evalEpisodes, ep] };
    }),

  loadEvalStatus: async () => {
    try {
      const result = await api.getEvalStatus();
      if (result.run) {
        set({ evalRun: result.run });
      }
    } catch {
      // silently fail
    }
  },

  // Navigate actions
  setNavStepDelay: (ms) => set({ navStepDelay: ms }),

  navPause: async () => {
    try {
      await api.navRunPause();
    } catch {
      // silently fail
    }
  },

  navStop: async () => {
    try {
      await api.navRunStop();
      set({ navStatus: "idle", navCurrentStep: 0 });
    } catch {
      // silently fail
    }
  },
}));

/** Wire WebSocket events to store updates. Call once at app startup. */
let _wsSubscribed = false;
export function subscribeWS() {
  if (_wsSubscribed) return;
  _wsSubscribed = true;
  wsManager.onStatusChange = (connected) => {
    useStore.getState().setConnected(connected);
    // On (re)connect: backfill any events the bus accumulated while we were down.
    if (connected) {
      void fetch("/api/errors")
        .then((r) => (r.ok ? r.json() : { events: [] }))
        .then((j) => {
          const events = (j?.events ?? []) as ErrorEnvelope[];
          if (events.length > 0) useErrorStore.getState().backfill(events);
        })
        .catch(() => {
          /* ignore — backfill is best-effort */
        });
    }
  };

  // Unified error/event channel — every backend bus envelope lands here.
  wsManager.on("error_event", (data) => {
    useErrorStore.getState().ingest(data as ErrorEnvelope);
  });

  // Graph dir changed on disk (e.g. a coding agent edited workspace/graphs/).
  // The Explorer registers a global refresh hook; nudge it to re-fetch.
  wsManager.on("graphs_changed", () => {
    const refresh = (window as unknown as Record<string, unknown>)
      .__refreshSavedGraphs;
    if (typeof refresh === "function") (refresh as () => void)();
  });

  wsManager.on("eval_progress", (data) => {
    useStore.getState().setEvalRun(data as EvalRunSummary);
  });

  wsManager.on("eval_episode_done", (data) => {
    useStore.getState().addOrUpdateEvalEpisode(data as EvalEpisodeResult);
  });

  wsManager.on("eval_complete", (data) => {
    useStore.getState().setEvalRun(data as EvalRunSummary);
  });

  // Navigate WS subscriptions — route by execution_id to per-node outputs
  const routeToNodes = (
    executionId: string | undefined,
    update: Partial<import("./types").NodeInstanceData>,
  ) => {
    if (!executionId) return;
    const bridge = getFlowStoreBridge();
    if (!bridge) return;
    const exec = bridge.getActiveExecution(executionId);
    if (!exec) return;
    bridge.updateNodeOutput(exec.agentNodeId, update);
    for (const nid of exec.outputNodeIds) {
      bridge.updateNodeOutput(nid, update);
    }
  };

  // Per-node viewer data — each viewer node emits its own WS event
  wsManager.on("viewer_data", (data) => {
    const { node_id, step, fields } = data as {
      node_id: string;
      step: number;
      fields: Record<string, unknown>;
    };
    const bridge = getFlowStoreBridge();
    if (!bridge) return;
    const prev = bridge.getNodeOutput(node_id);
    bridge.updateNodeOutput(node_id, {
      status: "running",
      currentStep: step,
      fields: { ...(prev?.fields || {}), ...fields },
    });
  });

  wsManager.on("nav_step", (data, raw) => {
    const step = data as NavStepData;
    const executionId = (raw as Record<string, unknown>).execution_id as
      | string
      | undefined;
    // Global store (backward compat for NavigatePage / EvalPage)
    useStore.setState((s) => ({
      navSteps: [...s.navSteps, step],
      navCurrentStep: step.step,
      navCurrentRgb: step.rgb_base64 || s.navCurrentRgb,
      navCurrentDepth: step.depth_base64 || s.navCurrentDepth,
    }));
    // Push state container live previews (home + nodeset-owned) into the
    // flow store for the bottom State panel.
    if (step.containers) {
      const bridge = getFlowStoreBridge();
      if (bridge) bridge.setContainersLive(step.containers);
    }
    // Route to agent node only (viewer nodes get data from viewer_data events)
    if (executionId) {
      const bridge = getFlowStoreBridge();
      if (bridge) {
        const exec = bridge.getActiveExecution(executionId);
        if (exec) {
          bridge.updateNodeOutput(exec.agentNodeId, {
            currentStep: step.step,
            currentRgb: step.rgb_base64 || "",
            currentDepth: step.depth_base64 || "",
            steps: [
              ...(bridge.getNodeOutput(exec.agentNodeId)?.steps || []),
              step,
            ],
            status: "running",
          });
        }
      }
    }
  });

  wsManager.on("nav_llm_step", (data, raw) => {
    const step = data as NavLLMStepData;
    const executionId = (raw as Record<string, unknown>).execution_id as
      | string
      | undefined;
    useStore.setState((s) => ({
      navLLMSteps: [...s.navLLMSteps, step],
      navCurrentStep: step.step,
      navCurrentRgb: step.rgb_base64 || s.navCurrentRgb,
      navCurrentDepth: step.depth_base64 || s.navCurrentDepth,
    }));
    if (executionId) {
      const bridge = getFlowStoreBridge();
      if (bridge) {
        const exec = bridge.getActiveExecution(executionId);
        if (exec) {
          bridge.updateNodeOutput(exec.agentNodeId, {
            currentStep: step.step,
            llmSteps: [
              ...(bridge.getNodeOutput(exec.agentNodeId)?.llmSteps || []),
              step,
            ],
            status: "running",
          });
        }
      }
    }
  });

  wsManager.on("nav_status", (data, raw) => {
    const d = data as Record<string, unknown>;
    const status = d.status as NavStatus;
    const executionId = (raw as Record<string, unknown>).execution_id as
      | string
      | undefined;
    useStore.setState({ navStatus: status });
    if (d.step !== undefined)
      useStore.setState({ navCurrentStep: d.step as number });
    routeToNodes(executionId, { status });
  });

  wsManager.on("nav_complete", (data, raw) => {
    const d = data as Record<string, unknown>;
    const executionId = (raw as Record<string, unknown>).execution_id as
      | string
      | undefined;
    const metrics = (d.metrics as Record<string, number>) || null;
    useStore.setState({
      navStatus: "done",
      navMetrics: metrics,
      navCurrentStep: (d.step as number) || useStore.getState().navCurrentStep,
    });
    routeToNodes(executionId, { status: "done", metrics });
    // (Auto-advance to the next episode is now an env panel concern —
    // an env panel may implement it inside on_action("play") or via a
    // dedicated action; the store no longer reaches into env state.)
  });

  wsManager.connect();
}
