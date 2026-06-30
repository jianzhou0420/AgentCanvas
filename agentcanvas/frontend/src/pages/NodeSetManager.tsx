/** NodeSet Manager page — two-column layout: Resources | Active.
 *
 * Left column (Resources): all discovered nodesets and servers.
 *   Unloaded items get a "Load" / "Start" button. Loaded items show an "Active" badge.
 *
 * Right column (Active): only currently loaded/connected items with full detail,
 *   tool tags, and Unload/Stop/Restart controls.
 *
 * See ADR-008.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw, Loader2, Package } from "lucide-react";
import { api } from "../api";
import type { ServerStatus } from "../api";
import type { NodeSetInfo } from "../types";
import NodeSetCard from "./manager/NodeSetCard";
import ServerCard from "./manager/ServerCard";

const SERVER_POLL_MS = 5000;

const noop = async () => {};

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
      {children}
    </h3>
  );
}

export default function NodeSetManager() {
  const [nodesets, setNodesets] = useState<NodeSetInfo[]>([]);
  const [servers, setServers] = useState<ServerStatus[]>([]);
  const [rescanning, setRescanning] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  // ── Fetch helpers ──

  const fetchNodesets = useCallback(async () => {
    try {
      setNodesets(await api.listNodesets());
    } catch {
      /* ignore */
    }
  }, []);

  const fetchServers = useCallback(async () => {
    try {
      setServers(await api.listServers());
    } catch {
      /* ignore */
    }
  }, []);

  const fetchAll = useCallback(async () => {
    await Promise.all([fetchNodesets(), fetchServers()]);
  }, [fetchNodesets, fetchServers]);

  // ── Initial load + server polling ──

  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchServers, SERVER_POLL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fetchAll, fetchServers]);

  // ── Rescan ──

  const handleRescan = async () => {
    setRescanning(true);
    try {
      await api.reloadComponents();
      await fetchAll();
    } finally {
      setRescanning(false);
    }
  };

  // ── NodeSet actions ──

  const handleLoadNodeset = (name: string) => async () => {
    await api.loadNodeset(name);
    await fetchNodesets();
  };

  const handleUnloadNodeset = (name: string) => async () => {
    await api.unloadNodeset(name);
    await fetchNodesets();
  };

  // ── Server actions ──

  const handleStartServer = (name: string) => async () => {
    await api.startServer(name);
    await fetchServers();
  };

  const handleStopServer = (name: string) => async () => {
    await api.stopServer(name);
    await fetchServers();
  };

  const handleRestartServer = (name: string) => async () => {
    await api.restartServer(name);
    await fetchServers();
  };

  // ── Derived active lists ──

  const localNodesets = nodesets.filter((ns) => !ns.requires_server);
  const serverNodesets = nodesets.filter((ns) => ns.requires_server);
  const activeLocalNodesets = localNodesets.filter((ns) => ns.loaded);
  const activeServerNodesets = serverNodesets.filter((ns) => ns.loaded);
  const activeServers = servers.filter(
    (srv) => srv.connected || srv.status === "starting",
  );
  const hasActive =
    activeLocalNodesets.length > 0 ||
    activeServerNodesets.length > 0 ||
    activeServers.length > 0;

  return (
    <div className="flex h-full w-full flex-col bg-gray-950">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-gray-800 px-6 py-3">
        <div>
          <h1 className="text-xl font-bold text-gray-100">NodeSet Manager</h1>
          <p className="text-sm text-gray-500">
            Manage local and server-mode NodeSets.
          </p>
        </div>
        <button
          onClick={handleRescan}
          disabled={rescanning}
          className="flex items-center gap-2 rounded bg-gray-800 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 disabled:opacity-50"
        >
          {rescanning ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <RefreshCw size={14} />
          )}
          Rescan workspace/
        </button>
      </div>

      {/* Two-column body */}
      <div className="flex flex-1 min-h-0">
        {/* ── Left: Resources ── */}
        <div className="flex w-1/2 flex-col border-r border-gray-800">
          <div className="border-b border-gray-800 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Resources
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Local Mode */}
            <section>
              <SectionHeading>Local Mode</SectionHeading>
              <div className="space-y-2">
                {localNodesets.length === 0 && (
                  <div className="rounded border border-gray-800 bg-gray-900/50 p-3 text-sm text-gray-600">
                    No local nodesets discovered
                  </div>
                )}
                {localNodesets.map((ns) => (
                  <NodeSetCard
                    key={ns.name}
                    variant="resource"
                    name={ns.name}
                    description={ns.description}
                    loaded={ns.loaded}
                    tools={ns.tools}
                    onLoad={handleLoadNodeset(ns.name)}
                    onUnload={noop}
                  />
                ))}
              </div>
            </section>

            {/* Server Mode */}
            <section>
              <SectionHeading>Server Mode</SectionHeading>
              <div className="space-y-2">
                {serverNodesets.length === 0 && servers.length === 0 && (
                  <div className="rounded border border-gray-800 bg-gray-900/50 p-3 text-sm text-gray-600">
                    No server-mode nodesets discovered
                  </div>
                )}
                {serverNodesets.map((ns) => (
                  <NodeSetCard
                    key={ns.name}
                    variant="resource"
                    name={ns.name}
                    description={ns.description}
                    loaded={ns.loaded}
                    tools={ns.tools}
                    onLoad={handleLoadNodeset(ns.name)}
                    onUnload={noop}
                  />
                ))}
                {servers.map((srv) => (
                  <ServerCard
                    key={srv.name}
                    variant="resource"
                    server={srv}
                    onStart={handleStartServer(srv.name)}
                    onStop={noop}
                    onRestart={noop}
                  />
                ))}
              </div>
            </section>
          </div>
        </div>

        {/* ── Right: Active ── */}
        <div className="flex w-1/2 flex-col">
          <div className="border-b border-gray-800 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Active
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {!hasActive ? (
              <div className="flex h-full items-center justify-center">
                <div className="text-center">
                  <Package size={32} className="mx-auto mb-2 text-gray-700" />
                  <div className="text-sm text-gray-500">
                    No active resources
                  </div>
                  <div className="mt-1 text-xs text-gray-600">
                    Load a nodeset from Resources
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Active Local Mode */}
                {activeLocalNodesets.length > 0 && (
                  <section>
                    <SectionHeading>Local Mode</SectionHeading>
                    <div className="space-y-2">
                      {activeLocalNodesets.map((ns) => (
                        <NodeSetCard
                          key={ns.name}
                          variant="active"
                          name={ns.name}
                          description={ns.description}
                          loaded={ns.loaded}
                          tools={ns.tools}
                          onLoad={noop}
                          onUnload={handleUnloadNodeset(ns.name)}
                        />
                      ))}
                    </div>
                  </section>
                )}

                {/* Active Server Mode */}
                {(activeServerNodesets.length > 0 ||
                  activeServers.length > 0) && (
                  <section>
                    <SectionHeading>Server Mode</SectionHeading>
                    <div className="space-y-2">
                      {activeServerNodesets.map((ns) => (
                        <NodeSetCard
                          key={ns.name}
                          variant="active"
                          name={ns.name}
                          description={ns.description}
                          loaded={ns.loaded}
                          tools={ns.tools}
                          onLoad={noop}
                          onUnload={handleUnloadNodeset(ns.name)}
                        />
                      ))}
                      {activeServers.map((srv) => (
                        <ServerCard
                          key={srv.name}
                          variant="active"
                          server={srv}
                          onStart={noop}
                          onStop={handleStopServer(srv.name)}
                          onRestart={handleRestartServer(srv.name)}
                        />
                      ))}
                    </div>
                  </section>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
