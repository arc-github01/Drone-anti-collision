import { displayAttitude, horizonOffsetPx, horizonRotationDeg } from "../attitude";
import { Playback, usePlayback } from "../playback";

/** One-decimal degrees without a misleading "-0.0". */
const deg1 = (v: number) => (v.toFixed(1) === "-0.0" ? "0.0" : v.toFixed(1)) + "°";
import type { Replay, ReplayState } from "../replay";
import type { Recording } from "../types";

export type Status = "COLLISION" | "MARGIN VIOLATED" | "AVOIDING" | "RETURNING" | "CLEARED" | "CRUISE";

/** Status derived only from recorded state: clearance, lateral deviation from the original line, progress past the panel. */
export function statusOf(s: ReplayState, rec: Recording): Status {
  const ob = rec.environment.obstacles[0];
  const dev = Math.hypot(s.position[1] - rec.start_m[1], s.position[2] - rec.start_m[2]);
  const pastPanel = s.position[0] > ob.center_m[0] + ob.half_extents_m[0];
  if (s.clearance < 0) return "COLLISION";
  if (s.clearance < rec.environment.safety_margin_m) return "MARGIN VIOLATED";
  if (dev > 0.05) return pastPanel ? "RETURNING" : "AVOIDING";
  return pastPanel ? "CLEARED" : "CRUISE";
}

const TONE: Record<Status, string> = {
  COLLISION: "bad",
  "MARGIN VIOLATED": "warn",
  AVOIDING: "info",
  RETURNING: "info",
  CLEARED: "ok",
  CRUISE: "neutral",
};

/** rollRight: + = banked right. pitchUp: + = nose up (more sky, horizon moves down). */
function Horizon({ rollRight, pitchUp }: { rollRight: number; pitchUp: number }) {
  const off = horizonOffsetPx(pitchUp);
  const roll = horizonRotationDeg(rollRight);
  return (
    <svg width="92" height="92" viewBox="-46 -46 92 92" aria-label="attitude indicator">
      <defs>
        <clipPath id="hz">
          <circle r="44" />
        </clipPath>
      </defs>
      <g clipPath="url(#hz)">
        <g transform={`rotate(${roll}) translate(0 ${off})`}>
          <rect x="-90" y="-90" width="180" height="90" fill="#2f6fb3" />
          <rect x="-90" y="0" width="180" height="90" fill="#7a5a34" />
          <line x1="-90" x2="90" y1="0" y2="0" stroke="#fff" strokeWidth="1.2" />
        </g>
      </g>
      <circle r="44" fill="none" stroke="#9aa4b8" strokeWidth="1.5" />
      <path d="M-18 0 H-6 M6 0 H18 M0 -3 V3" stroke="#ffd166" strokeWidth="2" />
    </svg>
  );
}

interface Props {
  rec: Recording;
  replay: Replay;
  pb: Playback;
  label: string;
}

export function Hud({ rec, replay, pb, label }: Props) {
  usePlayback(pb);
  const s = replay.at(pb.time);
  const st = statusOf(s, rec);
  const speed = Math.hypot(...s.velocity);
  const { rollRightDeg: roll, pitchUpDeg: pitch, yawDeg: yaw } = displayAttitude(s.eulerDeg);
  return (
    <div className="hud" data-testid="hud">
      <div className={`status ${TONE[st]}`} data-testid="status">
        {st}
      </div>
      <div className="hud-flight">{label}</div>
      <div className="hud-row">
        <Horizon rollRight={roll} pitchUp={pitch} />
        <dl>
          <dt title="positive = banked right">Roll →</dt>
          <dd>{deg1(roll)}</dd>
          <dt title="positive = nose up">Pitch ↑</dt>
          <dd>{deg1(pitch)}</dd>
          <dt>Yaw</dt>
          <dd>{deg1(yaw)}</dd>
        </dl>
        <dl>
          <dt>Speed</dt>
          <dd>{speed.toFixed(2)} m/s</dd>
          <dt>v (x,y,z)</dt>
          <dd>
            {s.velocity.map((v) => v.toFixed(2)).join(", ")}
          </dd>
          <dt>Altitude</dt>
          <dd>{s.position[2].toFixed(2)} m</dd>
        </dl>
        <dl>
          <dt>Clearance</dt>
          <dd className={s.clearance < 0 ? "bad" : s.clearance < rec.environment.safety_margin_m ? "warn" : "ok"}>{s.clearance.toFixed(3)} m</dd>
          <dt>Panel ahead</dt>
          <dd>{Math.max(0, rec.environment.obstacles[0].center_m[0] - s.position[0]).toFixed(2)} m</dd>
          <dt>Motors sat.</dt>
          <dd>{s.saturated ? "yes" : "no"}</dd>
        </dl>
        <dl data-testid="hud-wind">
          <dt>Wind</dt>
          <dd>{Math.hypot(...s.wind).toFixed(1)} m/s</dd>
          <dt>w (x,y,z)</dt>
          <dd>{s.wind.map((v) => v.toFixed(1)).join(", ")}</dd>
          <dt>Airspeed</dt>
          <dd>{s.airspeed.toFixed(2)} m/s</dd>
          <dt>Tilt/accel limit</dt>
          <dd className={s.limited ? "warn" : ""}>{s.limited ? "ACTIVE" : "no"}</dd>
        </dl>
      </div>
    </div>
  );
}
