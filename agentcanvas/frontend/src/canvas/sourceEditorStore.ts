/** Source-editor notification state.
 *
 * Mirrors the backend watcher's `components_changed` WS broadcasts
 * (subscribed in store.ts) so the Source tab can flip "Saved" →
 * "Reloaded ✓" honestly and raise the stale-server banner.
 */

import { create } from "zustand";

interface SourceEditorState {
  /** Bumped on every components_changed broadcast from the watcher. */
  changeSeq: number;
  lastReloaded: string[];
  lastStale: string[];
  noteComponentsChanged: (reloaded: string[], stale: string[]) => void;
}

export const useSourceEditorStore = create<SourceEditorState>((set) => ({
  changeSeq: 0,
  lastReloaded: [],
  lastStale: [],
  noteComponentsChanged: (reloaded, stale) =>
    set((s) => ({
      changeSeq: s.changeSeq + 1,
      lastReloaded: reloaded,
      lastStale: stale,
    })),
}));
