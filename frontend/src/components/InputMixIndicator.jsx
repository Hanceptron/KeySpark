import { keyboardMouseMix } from "../utils";

// Thin horizontal band showing the keyboard-vs-mouse mix for the
// most recent 1-minute window. Marker on the far left means the
// latest minute was 100% keyboard input; on the far right means
// 100% mouse clicks. SVG glyphs (not emoji) bookend the track so
// screen reader rendering is consistent across platforms.
//
// When the latest window has zero of both, the marker is hidden and
// the right-side readout shows "no input" rather than 0% / 0% — the
// distinction matters for users skimming whether they were active
// at all in the last minute.
export function InputMixIndicator({ latest }) {
  const mix = keyboardMouseMix(latest); // 0..1 or null
  const kbPct = mix != null ? Math.round((1 - mix) * 100) : null;
  const msPct = mix != null ? Math.round(mix * 100) : null;
  const ariaLabel =
    mix != null
      ? `${kbPct}% keyboard, ${msPct}% mouse in the latest minute`
      : "No input recorded in the latest minute";

  return (
    <div
      role="img"
      aria-label={ariaLabel}
      className="glass-panel-sm flex items-center gap-3"
    >
      <KeyboardIcon className="w-4 h-4 text-zinc-400 shrink-0" />
      <div className="flex-1 relative h-2 glass-track rounded-full">
        {mix != null && (
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-zinc-100 rounded-full border-2 border-white/20 transition-all"
            style={{ left: `${mix * 100}%` }}
          />
        )}
      </div>
      <MouseIcon className="w-4 h-4 text-zinc-400 shrink-0" />
      <span className="text-xs text-zinc-500 tabular-nums w-28 text-right">
        {mix != null ? `${kbPct}% kb / ${msPct}% ms` : "no input"}
      </span>
    </div>
  );
}

function KeyboardIcon({ className }) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden="true">
      <rect
        x="1"
        y="4"
        width="14"
        height="8"
        rx="1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
      />
      <circle cx="4" cy="8" r="0.8" fill="currentColor" />
      <circle cx="8" cy="8" r="0.8" fill="currentColor" />
      <circle cx="12" cy="8" r="0.8" fill="currentColor" />
    </svg>
  );
}

function MouseIcon({ className }) {
  return (
    <svg viewBox="0 0 16 16" className={className} aria-hidden="true">
      <rect
        x="5"
        y="2"
        width="6"
        height="12"
        rx="3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
      />
      <line x1="8" y1="2" x2="8" y2="6" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}
