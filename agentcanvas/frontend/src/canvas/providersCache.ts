/** Module-level cache of the provider registry (id → ProviderInfo).
 *
 * One fetch serves every consumer (model chips, pickers) instead of one
 * request per canvas card. Invalidated by the "providers-changed" window
 * event, which Settings dispatches after key edits.
 */

import { useSyncExternalStore } from "react";
import { api } from "../api";
import type { ProvidersMap, ProviderInfo } from "../types";

let cache: ProvidersMap | null = null;
let inflight = false;
const listeners = new Set<() => void>();

function refetch() {
  if (inflight) return;
  inflight = true;
  api
    .getProviders()
    .then((p) => {
      cache = p;
      listeners.forEach((l) => l());
    })
    .catch(() => {
      /* backend unreachable — consumers keep the null/stale view */
    })
    .finally(() => {
      inflight = false;
    });
}

if (typeof window !== "undefined") {
  window.addEventListener("providers-changed", () => {
    cache = null;
    refetch();
  });
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (cache === null) refetch();
  return () => {
    listeners.delete(listener);
  };
}

/** Reactive provider registry — null until the first fetch resolves. */
export function useProviders(): ProvidersMap | null {
  return useSyncExternalStore(subscribe, () => cache);
}

/** Reverse-lookup a provider by its display label ("OpenAI" → "openai").
 * Profile option labels are "«provider label» / «model»", so this is how
 * chip code resolves a profile reference back to key status. */
export function providerByLabel(
  providers: ProvidersMap,
  label: string,
): [string, ProviderInfo] | null {
  for (const [id, info] of Object.entries(providers)) {
    if (info.label === label) return [id, info];
  }
  return null;
}
