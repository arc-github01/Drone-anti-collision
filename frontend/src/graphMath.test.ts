import { describe, expect, it } from "vitest";
import { decimate, ticks, timeToX, xToTime, yRange } from "./graphMath";

describe("yRange", () => {
  it("covers all series with padding", () => {
    const [lo, hi] = yRange([[0.2, 0.4], [0.3, 0.9]], 0.01);
    expect(lo).toBeLessThan(0.2);
    expect(hi).toBeGreaterThan(0.9);
  });
  it("enforces a minimum span around a (nearly) flat signal", () => {
    const [lo, hi] = yRange([[0.7, 0.7000001]], 0.02, 0);
    expect(hi - lo).toBeCloseTo(0.02, 9);
    expect((lo + hi) / 2).toBeCloseTo(0.7, 6);
  });
  it("falls back to [0,1] with no data", () => {
    expect(yRange([], 0.1)).toEqual([0, 1]);
  });
});

describe("decimate", () => {
  const N = 5000;
  const t = Array.from({ length: N }, (_, i) => i * 0.004);
  const v = t.map((x) => 0.7 + 0.01 * Math.sin(x * 3));
  it("returns the data unchanged when it is already small", () => {
    const d = decimate(t.slice(0, 50), v.slice(0, 50), 100);
    expect(d.t.length).toBe(50);
  });
  it("reduces to roughly two points per pixel column and keeps time ordered", () => {
    const d = decimate(t, v, 300);
    expect(d.t.length).toBeLessThanOrEqual(2 * 300 + 2);
    for (let i = 1; i < d.t.length; i++) expect(d.t[i]).toBeGreaterThanOrEqual(d.t[i - 1]);
  });
  it("never loses a short spike, and keeps the global min and max", () => {
    const spiky = v.slice();
    spiky[2503] = 1.0; // one-sample spike
    spiky[1201] = 0.0;
    const d = decimate(t, spiky, 120);
    expect(Math.max(...d.v)).toBe(1.0);
    expect(Math.min(...d.v)).toBe(0.0);
  });
});

describe("axis mapping", () => {
  it("timeToX and xToTime are inverses and clamp", () => {
    for (const tt of [0, 1.3, 7.77, 10]) expect(xToTime(timeToX(tt, 0, 10, 40, 940), 0, 10, 40, 940)).toBeCloseTo(tt, 9);
    expect(xToTime(-100, 0, 10, 40, 940)).toBe(0);
    expect(xToTime(5000, 0, 10, 40, 940)).toBe(10);
  });
  it("ticks are evenly spaced, inside the range", () => {
    const k = ticks(0, 14.7, 5);
    expect(k.length).toBeGreaterThan(2);
    expect(k.every((x) => x >= 0 && x <= 14.7)).toBe(true);
    const steps = k.slice(1).map((x, i) => +(x - k[i]).toFixed(9));
    expect(new Set(steps).size).toBe(1);
  });
});
