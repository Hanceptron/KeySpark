import { useEffect, useState } from "react";

// Fetch `url` once on mount and then every `intervalMs`. Returns
// { data, failing, lastSuccessMs }:
//   - data: the most recent successful JSON response, or null until the
//     first success. Kept across later failures so a brief backend blip
//     does not blank the dashboard.
//   - failing: true once >= 2 consecutive fetches have failed, so the UI
//     can show a distinct "reconnecting" state instead of silently aging
//     stale data into a misleading "offline". One dropped poll does not
//     trip it (avoids flicker on a single blip).
//   - lastSuccessMs: Date.now() of the last successful fetch, or null.
// Errors are logged to the console in dev. The cleanup cancels the
// interval and ignores any in-flight response on unmount / dep change.
export function usePolling(url, intervalMs) {
  const [state, setState] = useState({
    data: null,
    failing: false,
    lastSuccessMs: null,
  });

  useEffect(() => {
    // A null/empty url means "nothing to fetch yet" (e.g. a day panel
    // before its user is known). Skip cleanly rather than fetching
    // "/null" and spamming 404s.
    if (!url) return undefined;

    let cancelled = false;
    let failures = 0;

    async function fetchOnce() {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status} for ${url}`);
        }
        const json = await res.json();
        failures = 0;
        if (!cancelled) {
          setState({ data: json, failing: false, lastSuccessMs: Date.now() });
        }
      } catch (err) {
        failures += 1;
        if (!cancelled) {
          console.warn(`usePolling: ${err.message ?? err}`);
          // Keep the last data + lastSuccessMs; only flip `failing` after
          // two misses in a row so a single transient error is ignored.
          setState((prev) => ({ ...prev, failing: failures >= 2 }));
        }
      }
    }

    fetchOnce();
    const id = setInterval(fetchOnce, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [url, intervalMs]);

  return state;
}
