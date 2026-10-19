import { useEffect, useState } from "react";

// Re-render on a fixed wall-clock interval so a component can read a
// fresh `now` without calling Date.now() during render (the
// react-hooks/purity rule rejects impure calls in the render body).
// Lazy init + the interval callback keep the impure call out of render.
export function useNow(intervalMs = 30_000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
