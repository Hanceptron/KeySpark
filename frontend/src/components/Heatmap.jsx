import { useMemo } from "react";

// A true cold-to-hot heatmap of mouse traffic. One blurred cell per
// grid square from the API, filtered to the chosen event type
// ("move" or "click"). Unlike a flat single-color map, each cell is
// colored by its normalized intensity along a blue -> cyan -> green
// -> yellow -> red ramp, so dense "hot zones" read as red and sparse
// activity as cool blue. A mild Gaussian blur merges the discrete
// grid into continuous zones; hotter cells are painted last so they
// sit on top of their cooler neighbours.
//
// Intensity uses log(count+1)/log(max+1) so a cell hit 5 times is
// still visible next to one hit 5000 times, while the gradient still
// spans the full cold-to-hot range. Each panel is normalized to its
// own max, so the Movement and Clicks maps each surface their own
// hot zones regardless of absolute volume.

// Cold -> hot color ramp. t in [0,1] -> CSS rgb string. Stops
// approximate a perceptual "turbo"/jet gradient.
const HEAT_STOPS = [
  [0.0, [12, 28, 110]], // deep indigo (cold)
  [0.2, [20, 120, 220]], // blue
  [0.4, [0, 200, 205]], // cyan
  [0.55, [55, 200, 95]], // green
  [0.7, [240, 225, 45]], // yellow
  [0.85, [242, 140, 30]], // orange
  [1.0, [231, 33, 33]], // hot red
];

function heatColor(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const [t1, c1] = HEAT_STOPS[i];
    if (x <= t1) {
      const [t0, c0] = HEAT_STOPS[i - 1];
      const f = (x - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(231,33,33)";
}

// CSS gradient string for the legend bar, built from the same stops.
const LEGEND_GRADIENT = `linear-gradient(to right, ${HEAT_STOPS.map(
  ([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${Math.round(t * 100)}%`,
).join(", ")})`;

export function Heatmap({ data, type, title, frameW = 108, frameH = 70 }) {
  const { cells, maxCount, denom } = useMemo(() => {
    // Clip to the primary-screen frame (drops the external-monitor
    // tail and any off-screen/negative coords), then sort ascending by
    // count so the hottest cells render last and sit on top.
    const cells = (data || [])
      .filter(
        (c) =>
          c.type === type &&
          c.cell_x >= 0 &&
          c.cell_x < frameW &&
          c.cell_y >= 0 &&
          c.cell_y < frameH,
      )
      .slice()
      .sort((a, b) => (a.count ?? 0) - (b.count ?? 0));
    const maxCount = cells.reduce((m, c) => Math.max(m, c.count ?? 0), 0);
    return { cells, maxCount, denom: Math.log(maxCount + 1) };
  }, [data, type, frameW, frameH]);

  return (
    <div className="glass-panel">
      <div className="flex items-baseline justify-between mb-3 gap-3">
        <h2 className="text-sm text-zinc-400">{title}</h2>
        {maxCount > 0 && (
          <span className="text-xs text-zinc-500 tabular-nums">
            peak {maxCount}
          </span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${frameW} ${frameH}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full rounded"
        style={{ background: "#09090b" }}
        shapeRendering="crispEdges"
      >
        {/* Crisp 1x1 grid-aligned cells, no blur - a true pixel-style
            screen heatmap. The screen-shaped frame keeps the aspect
            correct regardless of monitor setup. */}
        {cells.map((c) => {
          const t = denom > 0 ? Math.log((c.count ?? 0) + 1) / denom : 0;
          return (
            <rect
              key={`${c.cell_x}-${c.cell_y}`}
              x={c.cell_x}
              y={c.cell_y}
              width={1}
              height={1}
              fill={heatColor(t)}
            >
              <title>{`(${c.cell_x}, ${c.cell_y}) - ${c.count}`}</title>
            </rect>
          );
        })}
      </svg>

      {cells.length === 0 ? (
        <div className="text-xs text-zinc-500 mt-2">
          no {type} events yet - move the mouse
        </div>
      ) : (
        <div className="flex items-center gap-2 mt-3">
          <span className="text-[10px] text-zinc-500">less</span>
          <div
            className="h-2 flex-1 rounded"
            style={{ background: LEGEND_GRADIENT }}
            role="img"
            aria-label="heatmap scale from low (blue) to high (red)"
          />
          <span className="text-[10px] text-zinc-500">more</span>
        </div>
      )}
    </div>
  );
}
