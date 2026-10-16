import { useState } from "react";
import { Playback, usePlayback } from "../playback";
import type { Recording, SimulateRequest, Summary } from "../types";
import { WIND_PRESETS, WindDraft, approachComponents, validateWind, windVector } from "../windMath";

export function Transport({ pb }: { pb: Playback }) {
  usePlayback(pb);
  return (
    <div className="transport" data-testid="transport">
      <button onClick={() => pb.toggle()} aria-label={pb.playing ? "Pause" : "Play"} data-testid="play">
        {pb.playing ? "❚❚" : "▶"}
      </button>
      <button onClick={() => pb.step(-0.1)} aria-label="Step back 0.1 s" title="−0.1 s">
        ⏮
      </button>
      <button onClick={() => pb.step(0.1)} aria-label="Step forward 0.1 s" title="+0.1 s">
        ⏭
      </button>
      <input
        type="range"
        min={0}
        max={pb.duration || 1}
        step={0.004}
        value={pb.time}
        onChange={(e) => {
          pb.pause();
          pb.seek(+e.target.value);
        }}
        aria-label="Simulation time"
        data-testid="scrubber"
      />
      <span className="time" data-testid="time">
        {pb.time.toFixed(2)} / {pb.duration.toFixed(2)} s
      </span>
      <select value={pb.speed} onChange={(e) => pb.setSpeed(+e.target.value)} aria-label="Playback speed">
        {[0.25, 0.5, 1, 2, 4].map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
      <label className="check">
        <input type="checkbox" checked={pb.loop} onChange={(e) => pb.setLoop(e.target.checked)} /> loop
      </label>
    </div>
  );
}

const num = (s: string) => (s.trim() === "" ? undefined : Number(s));

const f1 = (n: number) => (Math.abs(n) < 0.05 ? "0.0" : n.toFixed(1));

function WindControls({ wind, onWind }: { wind: WindDraft; onWind: (w: WindDraft) => void }) {
  const set = (k: keyof WindDraft) => (e: React.ChangeEvent<HTMLInputElement>) => onWind({ ...wind, [k]: e.target.value === "" ? NaN : Number(e.target.value) });
  const v = windVector(wind.speed, wind.azimuth, wind.elevation);
  const c = approachComponents(v);
  const err = validateWind(wind);
  return (
    <fieldset className="wind" data-testid="wind-controls">
      <legend>Wind</legend>
      <div className="presets">
        {WIND_PRESETS.map((p) => (
          <button key={p.label} type="button" onClick={() => onWind({ ...wind, speed: wind.speed > 0 ? wind.speed : 4, ...p.wind })}>
            {p.label}
          </button>
        ))}
        <button type="button" onClick={() => onWind({ ...wind, speed: 0 })} data-testid="wind-off">
          No wind
        </button>
      </div>
      <div className="form">
        <label>
          Speed (m/s)
          <input type="number" min={0} max={30} step={0.5} value={Number.isNaN(wind.speed) ? "" : wind.speed} onChange={set("speed")} data-testid="wind-speed" />
        </label>
        <label>
          Turbulence σ (m/s)
          <input type="number" min={0} max={5} step={0.1} value={Number.isNaN(wind.sigma) ? "" : wind.sigma} onChange={set("sigma")} data-testid="wind-sigma" />
        </label>
        <label className="wide">
          Azimuth (°) — horizontal direction it blows toward
          <span className="pair">
            <input type="range" min={-180} max={180} step={1} value={Number.isNaN(wind.azimuth) ? 0 : wind.azimuth} onChange={set("azimuth")} />
            <input type="number" min={-360} max={360} step={1} value={Number.isNaN(wind.azimuth) ? "" : wind.azimuth} onChange={set("azimuth")} data-testid="wind-azimuth" />
          </span>
        </label>
        <label className="wide">
          Elevation (°) — angle above horizontal
          <span className="pair">
            <input type="range" min={-90} max={90} step={1} value={Number.isNaN(wind.elevation) ? 0 : wind.elevation} onChange={set("elevation")} />
            <input type="number" min={-90} max={90} step={1} value={Number.isNaN(wind.elevation) ? "" : wind.elevation} onChange={set("elevation")} data-testid="wind-elevation" />
          </span>
        </label>
        {wind.sigma > 0 && (
          <>
            <label>
              Gust time (s)
              <input type="number" min={0.2} max={30} step={0.5} value={Number.isNaN(wind.tau) ? "" : wind.tau} onChange={set("tau")} />
            </label>
            <label>
              Gust seed
              <input type="number" min={0} step={1} value={Number.isNaN(wind.seed) ? "" : wind.seed} onChange={set("seed")} />
            </label>
          </>
        )}
      </div>
      <p className="muted small" data-testid="wind-readout">
        Vector (x fwd, y left, z up): <b>{f1(v[0])}, {f1(v[1])}, {f1(v[2])} m/s</b>
        <br />
        Headwind {f1(c.headwind)} · crosswind (left +) {f1(c.crosswindLeft)} · vertical (up +) {f1(c.vertical)} m/s
      </p>
      <p className="muted small">
        Azimuth 0° = tailwind, ±180° = headwind, +90° = toward the drone&apos;s left. Elevation +90° = updraft. The direction is where the wind blows <i>toward</i>.
      </p>
      {err && (
        <p className="error" role="alert" data-testid="wind-error">
          {err}
        </p>
      )}
    </fieldset>
  );
}

export function SimPanel({
  busy,
  onSimulate,
  onReset,
  modified,
  error,
  wind,
  onWind,
  windStale,
}: {
  busy: boolean;
  onSimulate: (r: SimulateRequest) => void;
  onReset: () => void;
  modified: boolean;
  error: string | null;
  wind: WindDraft;
  onWind: (w: WindDraft) => void;
  windStale: boolean;
}) {
  const [speed, setSpeed] = useState("");
  const [dodge, setDodge] = useState<NonNullable<SimulateRequest["dodge"]>>("auto");
  const [margin, setMargin] = useState("");
  const [altitude, setAltitude] = useState("");
  const [floor, setFloor] = useState(true);
  const [save, setSave] = useState(false);
  // Drone physics overrides — undefined means "use simulation default"
  const DEFAULT_MASS = 0.080;    // kg  (DroneParams default)
  const DEFAULT_THRUST = 0.40;   // N   (DroneParams default)
  const [mass, setMass] = useState<number>(DEFAULT_MASS);
  const [thrust, setThrust] = useState<number>(DEFAULT_THRUST);
  const massDefault = mass === DEFAULT_MASS;
  const thrustDefault = thrust === DEFAULT_THRUST;
  return (
    <section className="card" data-testid="sim-panel">
      <h2>Re-simulate</h2>
      <p className="muted small">Runs planning + controller + physics on the server from the stored obstacle data (no YOLO).</p>
      <div className="form">
        <label>
          Approach speed (m/s)
          <input value={speed} onChange={(e) => setSpeed(e.target.value)} placeholder="from video" inputMode="decimal" />
        </label>
        <label>
          Dodge direction
          <select value={dodge} onChange={(e) => setDodge(e.target.value as typeof dodge)}>
            {["auto", "left", "right", "up", "down"].map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </label>
        <label>
          Safety margin (m)
          <input value={margin} onChange={(e) => setMargin(e.target.value)} placeholder="0.25" inputMode="decimal" />
        </label>
        <label>
          Start altitude (m)
          <input value={altitude} onChange={(e) => setAltitude(e.target.value)} placeholder="1.5" inputMode="decimal" />
        </label>
        {/* ---- Drone physics overrides ---- */}
        <label className="stepper-label">
          Drone weight (g)
          <span className="stepper">
            <button type="button" className="step-btn" onClick={() => setMass((m) => Math.max(0.010, +(m - 0.010).toFixed(3)))} aria-label="Decrease weight">−</button>
            <input
              type="number" min={10} max={5000} step={1}
              value={Math.round(mass * 1000)}
              onChange={(e) => setMass(Math.min(5, Math.max(0.01, Number(e.target.value) / 1000)))}
              data-testid="mass-input"
            />
            <button type="button" className="step-btn" onClick={() => setMass((m) => Math.min(5, +(m + 0.010).toFixed(3)))} aria-label="Increase weight">+</button>
            {!massDefault && <button type="button" className="reset-btn" onClick={() => setMass(DEFAULT_MASS)} title="Reset to default">↺</button>}
          </span>
          <span className="param-hint">default {Math.round(DEFAULT_MASS * 1000)} g</span>
        </label>
        <label className="stepper-label">
          Max thrust / motor (N)
          <span className="stepper">
            <button type="button" className="step-btn" onClick={() => setThrust((t) => Math.max(0.05, +(t - 0.05).toFixed(2)))} aria-label="Decrease thrust">−</button>
            <input
              type="number" min={0.05} max={5} step={0.05}
              value={thrust}
              onChange={(e) => setThrust(Math.min(5, Math.max(0.05, Number(e.target.value))))}
              data-testid="thrust-input"
            />
            <button type="button" className="step-btn" onClick={() => setThrust((t) => Math.min(5, +(t + 0.05).toFixed(2)))} aria-label="Increase thrust">+</button>
            {!thrustDefault && <button type="button" className="reset-btn" onClick={() => setThrust(DEFAULT_THRUST)} title="Reset to default">↺</button>}
          </span>
          <span className="param-hint">default {DEFAULT_THRUST.toFixed(2)} N</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={floor} onChange={(e) => setFloor(e.target.checked)} /> floor at z = 0
        </label>
        <label className="check">
          <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} /> save as run recording
        </label>
      </div>
      <WindControls wind={wind} onWind={onWind} />
      {windStale && (
        <p className="warn small" data-testid="wind-stale">
          Wind settings changed — press Re-simulate to apply them.
        </p>
      )}
      <div className="btns">
        <button
          className="primary"
          disabled={busy || validateWind(wind) !== null}
          onClick={() =>
            onSimulate({
              speed_mps: num(speed),
              dodge,
              margin_m: num(margin),
              altitude_m: num(altitude),
              no_floor: !floor,
              mass_kg: massDefault ? undefined : mass,
              max_thrust_per_motor_n: thrustDefault ? undefined : thrust,
              wind_speed_mps: wind.speed,
              wind_azimuth_deg: wind.azimuth,
              wind_elevation_deg: wind.elevation,
              turbulence_sigma_mps: wind.sigma,
              turbulence_tau_s: wind.tau,
              wind_seed: wind.seed,
              save,
            })
          }
          data-testid="simulate"
        >
          {busy ? "Simulating…" : "Re-simulate"}
        </button>
        {modified && (
          <button onClick={onReset} disabled={busy}>
            Back to saved
          </button>
        )}
      </div>
      {error && (
        <p className="error" role="alert" data-testid="sim-error">
          {error}
        </p>
      )}
    </section>
  );
}

const row = (label: string, s: Summary | null) =>
  s ? (
    <tr key={label}>
      <th>{label}</th>
      <td className={s.collided ? "bad" : "ok"}>{s.min_clearance_m.toFixed(3)} m</td>
      <td>{s.collided ? "COLLISION" : "clear"}</td>
      <td>{s.max_tilt_deg.toFixed(1)}°</td>
      <td>{(s.max_tracking_error_m * 100).toFixed(1)} cm</td>
      <td className={(s.limited_fraction ?? 0) > 0.02 ? "warn" : ""}>{((s.limited_fraction ?? 0) * 100).toFixed(0)}%</td>
    </tr>
  ) : null;

export function PlanPanel({ rec }: { rec: Recording }) {
  const pl = rec.planning;
  const p = pl.plan;
  const dirName: Record<number, string> = { 0: "left", 45: "up-left", 90: "up", 135: "up-right", 180: "right", 225: "down-right", 270: "down", 315: "down-left" };
  return (
    <section className="card" data-testid="plan-panel">
      <h2>Avoidance plan</h2>
      <p className={p ? "" : "bad"} data-testid="plan-note">
        {pl.note}
      </p>
      {p && p.kind === "detour" && (
        <p>
          <b>Detour {dirName[p.params.direction_deg] ?? `${p.params.direction_deg}°`}</b> by {p.params.offset_m.toFixed(1)} m, {p.params.blend_s.toFixed(0)} s blend; planned
          min clearance {p.metrics.min_clearance_m.toFixed(2)} m, peak accel {p.metrics.peak_accel_mps2.toFixed(2)} m/s², peak tilt {p.metrics.peak_tilt_deg.toFixed(1)}°.
        </p>
      )}
      {p && p.kind === "brake" && <p>Braking stop over {p.params.stop_time_s.toFixed(1)} s (no detour was feasible).</p>}
      {rec.wind && (rec.wind.config.speed_mps > 0 || rec.wind.config.turbulence_sigma_mps > 0) && (
        <div className="windinfo" data-testid="wind-plan-info">
          <p>
            <b>Wind {rec.wind.config.speed_mps.toFixed(1)} m/s</b> toward azimuth {rec.wind.config.azimuth_deg.toFixed(0)}°, elevation {rec.wind.config.elevation_deg.toFixed(0)}°
            {rec.wind.config.turbulence_sigma_mps > 0 && ` + gusts σ ${rec.wind.config.turbulence_sigma_mps} m/s (seed ${rec.wind.config.seed})`}. Headwind{" "}
            {rec.wind.components_approach_frame.headwind_mps.toFixed(1)}, crosswind {rec.wind.components_approach_frame.crosswind_left_mps.toFixed(1)}, vertical{" "}
            {rec.wind.components_approach_frame.vertical_up_mps.toFixed(1)} m/s.
          </p>
          <p className={rec.wind.station_keeping.tilt_deg > Number(rec.drone.params.max_tilt_deg) * 0.8 ? "warn small" : "muted small"}>
            Holding still in this wind needs {rec.wind.station_keeping.tilt_deg.toFixed(1)}° tilt and {rec.wind.station_keeping.thrust_n.toFixed(2)} N (limit{" "}
            {String(rec.drone.params.max_tilt_deg)}°; weight {(Number(rec.drone.params.mass_kg) * 9.80665).toFixed(2)} N).
            {(rec.planning.attempts?.length ?? 0) > 1 && ` Re-planned ${(rec.planning.attempts?.length ?? 1) - 1}× because gusts reduced the flown clearance.`}
          </p>
        </div>
      )}
      {pl.n_candidates > 0 && (
        <p className="muted small">
          {pl.n_feasible} of {pl.n_candidates} candidate trajectories were feasible. Rejections (a candidate can fail several checks):{" "}
          {Object.entries(pl.rejections)
            .map(([k, v]) => `${k} ${v}`)
            .join(" · ")}
          .
        </p>
      )}
      <table className="summary">
        <thead>
          <tr>
            <th />
            <th>Min clear.</th>
            <th>Result</th>
            <th>Max tilt</th>
            <th>Track err.</th>
            <th title="share of the flight with the tilt/acceleration limit active">At limit</th>
          </tr>
        </thead>
        <tbody>
          {row("With avoidance", rec.summary)}
          {row("No avoidance", rec.nominal_summary)}
        </tbody>
      </table>
      <p className="muted small">Clearance is measured on the flown path (physics output), not on the plan.</p>
      <details>
        <summary>Assumptions &amp; uncertainty</summary>
        <ul className="small">
          {rec.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
          <li>
            Drone: mass {String(rec.drone.params.mass_kg)} kg, radius {rec.drone.radius_m} m, margin {rec.environment.safety_margin_m} m, start altitude is an assumed value.
          </li>
        </ul>
      </details>
    </section>
  );
}
