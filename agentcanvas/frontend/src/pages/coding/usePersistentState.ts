import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

/**
 * Drop-in `useState` whose value is mirrored to localStorage under `key`, so a
 * page refresh restores it instead of falling back to the initial value.
 *
 * Use ONLY for small UI selections that should feel sticky across a reload —
 * which sub-view/toggle/dropdown you last had open, form inputs. Never for
 * streamed log data or backend-fetched status (episode pass/fail marks): those
 * must come back fresh on reload, and the page's existing polling effects
 * refetch them once the restored selections are back in place.
 *
 * The setter keeps the full `useState` contract (functional updates included),
 * so this is a literal swap for `useState`.
 */
export function usePersistentState<T>(
  key: string,
  initial: T,
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw == null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage full or disabled — persistence is best-effort, never fatal */
    }
  }, [key, value]);

  return [value, setValue];
}
