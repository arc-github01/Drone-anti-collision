// Attitude as shown to the user (aviation convention) from the simulator's Euler angles.
//
// Simulator body frame is FLU (x forward, y left, z up) and angles are ZYX with right-handed rotations:
//   roll  about +x: positive = left side up = BANK RIGHT   (aviation: right bank positive  -> same sign)
//   pitch about +y: positive = NOSE DOWN                    (aviation: nose up positive    -> sign flipped)
//   yaw   about +z: positive = turning left (counter-clockwise from above)
import type { Vec3 } from "./types";

export interface DisplayAttitude {
  rollRightDeg: number; // + = banked right
  pitchUpDeg: number; // + = nose up
  yawDeg: number;
}

export function displayAttitude(eulerRpyDeg: Vec3): DisplayAttitude {
  const [roll, pitch, yaw] = eulerRpyDeg;
  return { rollRightDeg: roll + 0, pitchUpDeg: -pitch + 0, yawDeg: yaw + 0 }; // "+ 0" turns -0 into 0
}

/** Artificial-horizon geometry: nose up shows more sky, so the horizon line moves DOWN the instrument (SVG +y). */
export function horizonOffsetPx(pitchUpDeg: number, pxPerDeg = 1.4, limit = 28): number {
  return Math.max(-limit, Math.min(limit, pitchUpDeg * pxPerDeg));
}

/** Horizon rotation in SVG degrees (SVG positive = clockwise). Banking right makes the horizon appear to turn counter-clockwise. */
export function horizonRotationDeg(rollRightDeg: number): number {
  return -rollRightDeg;
}
