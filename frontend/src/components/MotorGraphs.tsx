import { memo, useEffect, useMemo, useRef, useState } from "react";
import { decimate, ticks, timeToX, xToTime, yRange } from "../graphMath";
import { Playback, usePlayback } from "../playback";
import type { Replay } from "../replay";
import type { MotorQuantity } from "../types";

export const MOTOR_COLORS = ["#4cc9f0", "#ffd166", "#7bd88f", "#ff8c69"];

const QUANTITIES: { key: MotorQuantity; label: string; unit: string; minSpan: number; digits: number }[] = [
  { key: "motor_command", label: "Command (controller output)", unit: "", minSpan: 0.01, digits: 4 },
  { key: "motor_omega_radps", label: "Angular speed", unit: "rad/s", minSpan: 10, digits: 1 },
  { key: "motor_thrust_n", label: "Thrust", unit: "N", minSpan: 0.01, digits: 4 },
  { key: "motor_power_w", label: "Electrical power", unit: "W", minSpan: 0.2, digits: 2 },
];

const H = 160;
const PAD = { l: 50, r: 70, t: 10, b: 22 };
const GAP = 10;
// Quadrant order = the drone seen from above, front at the top: M4 front-left, M1 front-right, M3 rear-left, M2 rear-right.
const QUADRANT_ORDER = [3, 0, 2, 1];

function useWidth(ref: React.RefObject<HTMLElement | null>) {
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(Math.max(260, Math.floor(e[0].contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

interface ChartProps {
  motor: number;
  name: string;
  t: Float64Array;
  values: Float64Array;
  width: number;
  yLo: number;
  yHi: number;
  color: string;
  unit: string;
  digits: number;
  baseline: number | null; // e.g. hover command
  baselineLabel: string;
  tMax: number;
  marker: number | null; // time of closest approach
  pb: Playback;
}

/** The static curve is memoised: only the playhead/readout re-renders while the clock runs. */
const Curve = memo(function Curve(p: Pick<ChartProps, "t" | "values" | "width" | "yLo" | "yHi" | "color" | "baseline" | "baselineLabel" | "tMax" | "marker" | "unit">) {
  const plotW = p.width - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const y = (v: number) => PAD.t + (1 - (v - p.yLo) / (p.yHi - p.yLo || 1)) * plotH;
  const x = (t: number) => timeToX(t, 0, p.tMax, PAD.l, PAD.l + plotW);
  const d = useMemo(() => decimate(p.t, p.values, plotW), [p.t, p.values, plotW]);
  const path = d.t.map((tt, i) => `${i ? "L" : "M"}${x(tt).toFixed(1)},${y(d.v[i]).toFixed(1)}`).join("");
  const yt = ticks(p.yLo, p.yHi, 3);
  const xt = ticks(0, p.tMax, Math.max(3, Math.floor(plotW / 90)));
  return (
    <g>
      {yt.map((v) => (
        <g key={`y${v}`}>
          <line x1={PAD.l} x2={PAD.l + plotW} y1={y(v)} y2={y(v)} className="grid" />
          <text x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="tick">
            {Math.abs(v) >= 100 ? v.toFixed(0) : +v.toPrecision(4)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={PAD.t} y2={PAD.t + plotH} className="grid" />
          <text x={x(t)} y={H - 6} textAnchor="middle" className="tick">
            {t}s
          </text>
        </g>
      ))}
      {p.baseline !== null && p.baseline >= p.yLo && p.baseline <= p.yHi && (
        <g>
          <line x1={PAD.l} x2={PAD.l + plotW} y1={y(p.baseline)} y2={y(p.baseline)} className="baseline" />
          <text x={PAD.l + plotW + 4} y={y(p.baseline) + 3} className="tick dim">
            {p.baselineLabel}
          </text>
        </g>
      )}
      {p.marker !== null && (
        <g>
          <line x1={x(p.marker)} x2={x(p.marker)} y1={PAD.t} y2={PAD.t + plotH} className="marker" />
          <text x={x(p.marker) + 3} y={PAD.t + 9} className="tick dim">
            closest
          </text>
        </g>
      )}
      <path d={path} fill="none" stroke={p.color} strokeWidth={1.6} strokeLinejoin="round" />
    </g>
  );
});

function Chart(p: ChartProps) {
  usePlayback(p.pb);
  const svgRef = useRef<SVGSVGElement>(null);
  const plotW = p.width - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const t = p.pb.time;
  const idx = Math.min(p.values.length - 1, Math.max(0, indexAt(p.t, t)));
  const val = p.values[idx];
  const px = timeToX(t, 0, p.tMax, PAD.l, PAD.l + plotW);
  const py = PAD.t + (1 - (val - p.yLo) / (p.yHi - p.yLo || 1)) * plotH;

  const seekFromEvent = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    p.pb.seek(xToTime(e.clientX - r.left, 0, p.tMax, PAD.l, PAD.l + plotW));
  };
  return (
    <svg
      ref={svgRef}
      className="chart"
      width={p.width}
      height={H}
      role="img"
      aria-label={`${p.name} versus simulation time`}
      data-testid={`motor-graph-${p.motor + 1}`}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        p.pb.pause();
        seekFromEvent(e);
      }}
      onPointerMove={(e) => e.buttons === 1 && seekFromEvent(e)}
    >
      <Curve {...p} />
      <text x={PAD.l + 6} y={PAD.t + 12} className="chart-title" fill={p.color}>
        {p.name}
      </text>
      <line x1={px} x2={px} y1={PAD.t} y2={PAD.t + plotH} className="playhead" />
      <circle cx={px} cy={Math.min(PAD.t + plotH, Math.max(PAD.t, py))} r={3.5} fill={p.color} className="playdot" />
      <text x={PAD.l + plotW + 8} y={PAD.t + 16} className="readout" fill={p.color}>
        {val.toFixed(p.digits)}
      </text>
      <text x={PAD.l + plotW + 8} y={PAD.t + 30} className="tick dim">
        {p.unit}
      </text>
    </svg>
  );
}

function indexAt(t: Float64Array, q: number) {
  let lo = 0;
  let hi = t.length - 1;
  if (q >= t[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= q) lo = mid;
    else hi = mid;
  }
  return lo;
}

interface Props {
  replay: Replay;
  pb: Playback;
  motorNames: string[];
  hoverCommand: number;
  closestApproachT: number | null;
}

export function MotorGraphs({ replay, pb, motorNames, hoverCommand, closestApproachT }: Props) {
  const [quantity, setQuantity] = useState<MotorQuantity>("motor_command");
  const [scale, setScale] = useState<"shared" | "each" | "full">("shared");
  const [delta, setDelta] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const fullWidth = useWidth(wrap);
  const twoCols = fullWidth >= 640; // otherwise one column
  const chartWidth = twoCols ? Math.floor((fullWidth - GAP) / 2) : fullWidth;
  const q = QUANTITIES.find((x) => x.key === quantity)!;
  const useDelta = delta && quantity === "motor_command";

  const series = useMemo(
    () =>
      [0, 1, 2, 3].map((m) => {
        const s = replay.motorSeries(m, quantity);
        return useDelta ? s.map((v) => v - hoverCommand) : s;
      }),
    [replay, quantity, useDelta, hoverCommand],
  );
  const shared = useMemo(
    () => (scale === "full" ? ([quantity === "motor_command" ? 0 : 0, quantity === "motor_command" ? 1 : Math.max(...series.map((s) => Math.max(...s))) * 1.05] as [number, number]) : yRange(series, q.minSpan)),
    [series, scale, q.minSpan, quantity],
  );

  return (
    <section className="graphs" data-testid="motor-graphs" aria-label="Motor graphs">
      <div className="graphs-head">
        <h2>Motors vs simulation time</h2>
        <label>
          Quantity
          <select value={quantity} onChange={(e) => setQuantity(e.target.value as MotorQuantity)}>
            {QUANTITIES.map((o) => (
              <option key={o.key} value={o.key}>
                {o.label}
                {o.unit ? ` (${o.unit})` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Y scale
          <select value={scale} onChange={(e) => setScale(e.target.value as typeof scale)}>
            <option value="shared">Zoom, shared by all four</option>
            <option value="each">Zoom, each its own</option>
            <option value="full">Full range</option>
          </select>
        </label>
        <label className={quantity === "motor_command" ? "" : "disabled"}>
          <input type="checkbox" checked={delta} disabled={quantity !== "motor_command"} onChange={(e) => setDelta(e.target.checked)} />
          Δ from hover
        </label>
      </div>
      <div ref={wrap} className="graph-stack" style={{ gridTemplateColumns: twoCols ? "1fr 1fr" : "1fr" }}>
        {QUADRANT_ORDER.map((m) => {
          const [lo, hi] = scale === "each" ? yRange([series[m]], q.minSpan) : shared;
          return (
            <div key={m} className="chart-card">
            <Chart
              motor={m}
              name={`Motor ${m + 1} · ${(motorNames[m] ?? "").replace(/^M\d\s*/, "")}`}
              t={replay.t}
              values={series[m]}
              width={chartWidth}
              yLo={lo}
              yHi={hi}
              color={MOTOR_COLORS[m]}
              unit={useDelta ? "Δ command" : q.unit || "command"}
              digits={useDelta ? 4 : q.digits}
              baseline={quantity === "motor_command" ? (useDelta ? 0 : hoverCommand) : null}
              baselineLabel="hover"
              tMax={replay.duration}
              marker={closestApproachT}
              pb={pb}
            />
            </div>
          );
        })}
      </div>
      <p className="graphs-note">
        Layout matches the drone seen from above (front at the top). Values are the controller's actual outputs from the recording. The four motors differ by very little in a gentle manoeuvre, so the y-axis is zoomed
        to the data; compare traces by the axis numbers, not by their height.
      </p>
    </section>
  );
}
