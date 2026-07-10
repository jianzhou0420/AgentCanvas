import { useEffect } from "react";
import { useStore, subscribeWS } from "./store";
import Header from "./components/Header";
import ErrorToast from "./components/ErrorToast";
import CanvasPage from "./canvas/CanvasPage";
import NodeSetManager from "./pages/NodeSetManager";
import EvalPage from "./eval/EvalPage";
import LogViewerPage from "./logs/LogViewerPage";
import ReplayPage from "./replay/ReplayPage";
import MonitorPage from "./pages/monitor/MonitorPage";

export default function App() {
  const loadEvalStatus = useStore((s) => s.loadEvalStatus);
  const appMode = useStore((s) => s.appMode);

  useEffect(() => {
    subscribeWS();
    loadEvalStatus();
  }, [loadEvalStatus]);

  return (
    <div className="h-screen overflow-hidden">
      <div id="app-header" style={{ height: 48 }}>
        <Header />
      </div>
      <div style={{ height: "calc(100vh - 48px)" }}>
        {appMode === "nav" && <CanvasPage />}
        {appMode === "manager" && <NodeSetManager />}
        {/* EvalPage stays mounted (just hidden) so its config — selected graph,
            selectors, episode/step counts — survives navigating away and back
            instead of resetting. `display: contents` removes this wrapper from
            layout when active, so the visible result is identical to rendering
            <EvalPage /> directly. State is still in-memory (lost on full page
            reload). */}
        <div style={{ display: appMode === "eval" ? "contents" : "none" }}>
          <EvalPage />
        </div>
        {appMode === "logs" && <LogViewerPage />}
        {appMode === "replay" && <ReplayPage />}
        {appMode === "monitor" && <MonitorPage />}
      </div>
      <ErrorToast />
    </div>
  );
}
