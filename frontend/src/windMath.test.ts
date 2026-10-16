import { describe, expect, it } from "vitest";
import { NO_WIND, WIND_PRESETS, approachComponents, validateWind, windAngles, windVector } from "./windMath";

const near = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-12) =>
  Array.from(a).forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol));

describe("windVector uses the documented 3D convention (direction the wind blows TOWARD)", () => {
  it("axis directions", () => {
    near(windVector(5, 0, 0), [5, 0, 0]); // tailwind
    near(windVector(5, 180, 0), [-5, 0, 0]); // headwind
    near(windVector(5, 90, 0), [0, 5, 0]); // toward the drone's left
    near(windVector(5, -90, 0), [0, -5, 0]); // toward the right
    near(windVector(5, 0, 90), [0, 0, 5]); // updraft
    near(windVector(5, 77, -90), [0, 0, -5]); // downdraft, azimuth irrelevant
  });
  it("general direction: magnitude, horizontal and vertical parts", () => {
    for (const [v, az, el] of [[3, 30, 20], [7.5, -120, -45], [1.2, 179, 80], [10, 0.5, 0.25]]) {
      const w = windVector(v, az, el);
      expect(Math.hypot(...w)).toBeCloseTo(v, 12);
      expect(Math.hypot(w[0], w[1])).toBeCloseTo(v * Math.cos((el * Math.PI) / 180), 12);
      expect(w[2]).toBeCloseTo(v * Math.sin((el * Math.PI) / 180), 12);
      const a = windAngles(w);
      expect(a.speed).toBeCloseTo(v, 9);
      expect(a.azimuthDeg).toBeCloseTo(az, 9);
      expect(a.elevationDeg).toBeCloseTo(el, 9);
    }
  });
  it("matches the Python reference values (dronesim.sim.wind.wind_vector)", () => {
    // wind_vector(4, 135, 30) and wind_vector(5, -60, -40) evaluated in Python
    near(windVector(4, 135, 30), [-2.449489742783178, 2.4494897427831783, 1.9999999999999998], 1e-12);
    near(windVector(5, -60, -40), [1.9151111077974456, -3.317069740844692, -3.2139380484326963], 1e-12);
  });
  it("zero and vertical edge cases for the inverse", () => {
    expect(windAngles([0, 0, 0])).toEqual({ speed: 0, azimuthDeg: 0, elevationDeg: 0 });
    expect(windAngles([0, 0, 4]).elevationDeg).toBeCloseTo(90, 12);
    expect(windAngles([0, 0, 4]).azimuthDeg).toBe(0);
  });
});

describe("approach-frame components", () => {
  it("headwind is positive against forward flight", () => {
    near(Object.values(approachComponents(windVector(6, 180, 0))), [6, 0, 0]);
    near(Object.values(approachComponents(windVector(6, 0, 0))), [-6, 0, 0]);
    const c = approachComponents(windVector(4, 90, 30));
    expect(c.crosswindLeft).toBeCloseTo(4 * Math.cos(Math.PI / 6), 12);
    expect(c.vertical).toBeCloseTo(2, 12);
    expect(Math.abs(c.headwind)).toBeLessThan(1e-12);
  });
});

describe("validation mirrors the API limits", () => {
  it("accepts a normal wind and the presets", () => {
    expect(validateWind({ ...NO_WIND, speed: 5, azimuth: 135, elevation: 30 })).toBeNull();
    for (const p of WIND_PRESETS) expect(validateWind({ ...NO_WIND, speed: 4, ...p.wind })).toBeNull();
  });
  it("rejects out-of-range and non-numeric values", () => {
    for (const bad of [{ speed: -1 }, { speed: 31 }, { elevation: 91 }, { elevation: -91 }, { sigma: 6 }, { tau: 0.1 }, { seed: -1 },
      { seed: 1.5 }, { azimuth: 400 }, { speed: NaN }]) {
      expect(validateWind({ ...NO_WIND, ...bad })).not.toBeNull();
    }
  });
});
