// Wind direction maths. Same convention as dronesim/sim/wind.py (kept in sync by windMath.test.ts):
//   world frame: +x = the drone's forward (approach) axis, +y = left, +z = up
//   azimuth   psi: in the horizontal plane from +x toward +y (counter-clockwise seen from above)
//   elevation eps: from the horizontal plane toward +z
//   w = V (cos eps cos psi, cos eps sin psi, sin eps)      -- the direction the wind blows TOWARD
import type { Vec3 } from "./types";

const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

export function windVector(speed: number, azimuthDeg: number, elevationDeg: number): Vec3 {
  const psi = rad(azimuthDeg);
  const eps = rad(elevationDeg);
  return [speed * Math.cos(eps) * Math.cos(psi), speed * Math.cos(eps) * Math.sin(psi), speed * Math.sin(eps)];
}

/** Inverse of windVector. Azimuth is 0 for a purely vertical or zero wind. */
export function windAngles(w: ArrayLike<number>): { speed: number; azimuthDeg: number; elevationDeg: number } {
  const speed = Math.hypot(w[0], w[1], w[2]);
  if (speed === 0) return { speed: 0, azimuthDeg: 0, elevationDeg: 0 };
  return {
    speed,
    azimuthDeg: Math.hypot(w[0], w[1]) === 0 ? 0 : deg(Math.atan2(w[1], w[0])),
    elevationDeg: deg(Math.asin(Math.min(1, Math.max(-1, w[2] / speed)))),
  };
}

/** Headwind is positive when the wind opposes forward flight (blows toward -x). */
export function approachComponents(w: ArrayLike<number>) {
  return { headwind: -w[0] + 0, crosswindLeft: w[1] + 0, vertical: w[2] + 0 };
}

export interface WindDraft {
  speed: number;
  azimuth: number;
  elevation: number;
  sigma: number;
  tau: number;
  seed: number;
}

export const NO_WIND: WindDraft = { speed: 0, azimuth: 0, elevation: 0, sigma: 0, tau: 2, seed: 0 };

export const WIND_PRESETS: { label: string; wind: Partial<WindDraft> }[] = [
  { label: "Headwind", wind: { azimuth: 180, elevation: 0 } },
  { label: "Tailwind", wind: { azimuth: 0, elevation: 0 } },
  { label: "From the right", wind: { azimuth: 90, elevation: 0 } },
  { label: "From the left", wind: { azimuth: -90, elevation: 0 } },
  { label: "Updraft", wind: { azimuth: 0, elevation: 90 } },
  { label: "Downdraft", wind: { azimuth: 0, elevation: -90 } },
];

/** Validate a draft; returns an error string or null. Limits match the API (dronesim/api/app.py). */
export function validateWind(w: WindDraft): string | null {
  const bad = (v: number) => !Number.isFinite(v);
  if ([w.speed, w.azimuth, w.elevation, w.sigma, w.tau, w.seed].some(bad)) return "Wind values must be numbers.";
  if (w.speed < 0 || w.speed > 30) return "Wind speed must be between 0 and 30 m/s.";
  if (w.azimuth < -360 || w.azimuth > 360) return "Azimuth must be between -360° and 360°.";
  if (w.elevation < -90 || w.elevation > 90) return "Elevation must be between -90° and 90°.";
  if (w.sigma < 0 || w.sigma > 5) return "Turbulence must be between 0 and 5 m/s.";
  if (w.tau < 0.2 || w.tau > 30) return "Gust correlation time must be between 0.2 and 30 s.";
  if (!Number.isInteger(w.seed) || w.seed < 0) return "Seed must be a non-negative integer.";
  return null;
}
