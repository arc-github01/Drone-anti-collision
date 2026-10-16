import { describe, expect, it } from "vitest";
import { displayAttitude, horizonOffsetPx, horizonRotationDeg } from "./attitude";
import { quatToEulerDeg } from "./replay";
import type { Vec3, Vec4 } from "./types";

// Rotate a vector by a body->world quaternion [w,x,y,z] (simulator frame).
const rotate = (q: Vec4, v: Vec3): Vec3 => {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
};
const aboutY = (deg: number): Vec4 => [Math.cos((deg * Math.PI) / 360), 0, Math.sin((deg * Math.PI) / 360), 0];
const aboutX = (deg: number): Vec4 => [Math.cos((deg * Math.PI) / 360), Math.sin((deg * Math.PI) / 360), 0, 0];

describe("pitch display follows the real nose direction", () => {
  it("a drone leaning back into a tailwind (simulator pitch -20) has its nose UP and shows +20 nose-up", () => {
    const q = aboutY(-20); // thrust (body z) tilts toward -x, i.e. against a tailwind
    const nose = rotate(q, [1, 0, 0]); // body forward axis in the world
    expect(nose[2]).toBeGreaterThan(0.3); // the nose really points up
    const thrust = rotate(q, [0, 0, 1]);
    expect(thrust[0]).toBeLessThan(-0.3); // and thrust leans backward
    const euler = quatToEulerDeg(q);
    expect(euler[1]).toBeCloseTo(-20, 9);
    expect(displayAttitude(euler).pitchUpDeg).toBeCloseTo(20, 9);
  });
  it("nose down (simulator pitch +20, leaning into a headwind) shows -20", () => {
    const q = aboutY(20);
    expect(rotate(q, [1, 0, 0])[2]).toBeLessThan(-0.3);
    expect(displayAttitude(quatToEulerDeg(q)).pitchUpDeg).toBeCloseTo(-20, 9);
  });
  it("sign of the displayed pitch always equals the sign of the nose's vertical component", () => {
    for (const p of [-35, -10, -1, 1, 10, 35]) {
      const q = aboutY(p);
      expect(Math.sign(displayAttitude(quatToEulerDeg(q)).pitchUpDeg)).toBe(Math.sign(rotate(q, [1, 0, 0])[2]));
    }
  });
  it("level shows exactly 0 (never -0)", () => {
    const a = displayAttitude([0, 0, 0]);
    expect(Object.is(a.pitchUpDeg, 0)).toBe(true);
    expect(Object.is(a.rollRightDeg, 0)).toBe(true);
  });
});

describe("roll display follows the real bank", () => {
  it("positive roll = right wing down = bank right", () => {
    const q = aboutX(15);
    const rightWing = rotate(q, [0, -1, 0]); // body right = -y
    expect(rightWing[2]).toBeLessThan(-0.2); // right wing is lower
    expect(displayAttitude(quatToEulerDeg(q)).rollRightDeg).toBeCloseTo(15, 9);
  });
});

describe("artificial horizon geometry", () => {
  it("nose up moves the horizon DOWN (more sky); nose down moves it up", () => {
    expect(horizonOffsetPx(20)).toBeGreaterThan(0);
    expect(horizonOffsetPx(-20)).toBeLessThan(0);
    expect(horizonOffsetPx(0)).toBe(0);
  });
  it("is clamped to the instrument", () => {
    expect(horizonOffsetPx(90)).toBe(28);
    expect(horizonOffsetPx(-90)).toBe(-28);
  });
  it("banking right turns the horizon counter-clockwise", () => {
    expect(horizonRotationDeg(10)).toBe(-10);
    expect(horizonRotationDeg(-10)).toBe(10);
  });
});
