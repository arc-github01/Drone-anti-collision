import { describe, expect, it } from "vitest";
import golden from "./test/fixtures/replay_golden.json";
import { Replay, videoTime } from "./replay";
import type { FlightBlock, Recording } from "./types";

const rec = golden.recording as unknown as Recording;
const flight = rec.flight as FlightBlock;
const replay = new Replay(flight);

const close = (a: ArrayLike<number> | number, b: ArrayLike<number> | number, tol = 1e-9) => {
  const A = typeof a === "number" ? [a] : Array.from(a);
  const B = typeof b === "number" ? [b] : Array.from(b);
  expect(A.length).toBe(B.length);
  A.forEach((v, i) => expect(Math.abs(v - B[i])).toBeLessThanOrEqual(tol));
};

describe("TypeScript Replay matches the Python reference implementation", () => {
  it("reproduces every golden query (positions, attitude, rates, all four motors, clearance)", () => {
    expect(golden.queries.length).toBeGreaterThan(40);
    for (const q of golden.queries) {
      const s = replay.at(q.t_query);
      expect(s.index).toBe(q.index);
      close(s.t, q.t);
      close(s.position, q.position_m);
      close(s.velocity, q.velocity_mps);
      close(s.quaternion, q.quaternion_wxyz, 1e-9);
      close(s.eulerDeg, q.euler_rpy_deg, 1e-6);
      close(s.bodyRates, q.body_rates_radps);
      close(s.motorCommand, q.motor_command);
      close(s.motorOmega, q.motor_omega_radps);
      close(s.motorThrust, q.motor_thrust_n);
      close(s.motorPower, q.motor_power_w);
      close(s.setpoint, q.setpoint_position_m);
      close(s.clearance, q.clearance_m);
      expect(s.saturated).toBe(q.saturated);
    }
  });
});

describe("replay semantics", () => {
  it("is exact at sample times and interpolates in between", () => {
    const i = 12;
    const s = replay.at(flight.t[i]);
    expect(s.index).toBe(i);
    close(s.position, flight.position_m[i]);
    const mid = replay.at((flight.t[i] + flight.t[i + 1]) / 2);
    close(mid.position, flight.position_m[i].map((v, k) => (v + flight.position_m[i + 1][k]) / 2));
    close(mid.motorCommand, flight.motor_command[i]); // zero-order hold, not interpolated
    expect(Math.hypot(...mid.quaternion)).toBeCloseTo(1, 12);
  });

  it("clamps outside the recorded range", () => {
    expect(replay.at(-10).index).toBe(0);
    expect(replay.at(1e6).index).toBe(replay.n - 1);
    expect(replay.at(1e6).t).toBeCloseTo(replay.duration, 12);
  });

  it("is order-independent: scrubbing forward, backward and shuffled gives identical states", () => {
    const ts = Array.from({ length: 41 }, (_, i) => (i / 40) * replay.duration);
    const fwd = ts.map((t) => replay.at(t));
    const back = [...ts].reverse().map((t) => replay.at(t)).reverse();
    const order = ts.map((_, i) => (i * 17) % 41); // a permutation of 0..40 (17 is coprime to 41)
    const shuffled: Record<number, ReturnType<Replay["at"]>> = {};
    order.forEach((i) => (shuffled[i] = replay.at(ts[i])));
    ts.forEach((_, i) => {
      expect(back[i]).toEqual(fwd[i]);
      expect(shuffled[i]).toEqual(fwd[i]);
    });
  });

  it("motorSeries returns exactly the controller outputs the graphs plot", () => {
    for (let m = 0; m < 4; m++) {
      const u = replay.motorSeries(m);
      expect(u.length).toBe(replay.n);
      close(u, flight.motor_command.map((r) => r[m]));
      close(replay.motorSeries(m, "motor_power_w"), flight.motor_power_w.map((r) => r[m]));
    }
  });

  it("pathUpTo grows with time and ends at the playhead sample", () => {
    expect(replay.pathUpTo(0).length).toBe(3);
    const t = replay.duration / 2;
    const p = replay.pathUpTo(t);
    expect(p.length).toBe((replay.indexAt(t) + 1) * 3);
    expect(replay.pathUpTo(1e9).length).toBe(replay.n * 3);
  });

  it("maps simulation time to video time, or null when unsynced", () => {
    expect(videoTime({ reference_video_time_s: 0.5, time_scale: 1 }, 2)).toBeCloseTo(2.5);
    expect(videoTime({ reference_video_time_s: 0.5, time_scale: 0.5 }, 2)).toBeCloseTo(1.5);
    expect(videoTime({ reference_video_time_s: 0.5, time_scale: null }, 2)).toBeNull();
  });
});

describe("attitude interpolation across the quaternion double cover", () => {
  // q and -q are the same rotation. A recording can hold consecutive samples with opposite signs;
  // interpolating them naively passes through ~zero and flips the drone. Replay must take the short way.
  const block = (qa: number[], qb: number[]): FlightBlock => {
    const v3 = [[0, 0, 0], [0, 0, 0]] as FlightBlock["position_m"];
    const v4 = [[0, 0, 0, 0], [0, 0, 0, 0]] as FlightBlock["motor_command"];
    return {
      t: [0, 1], position_m: v3, velocity_mps: v3, quaternion_wxyz: [qa, qb] as FlightBlock["quaternion_wxyz"],
      euler_rpy_deg: v3, body_rates_radps: v3, motor_command: v4, motor_omega_radps: v4, motor_thrust_n: v4,
      motor_power_w: v4, setpoint_position_m: v3, clearance_m: [1, 1], saturated: [false, false],
    };
  };
  const rollQuat = (deg: number, sign = 1) => {
    const h = (deg * Math.PI) / 360;
    return [sign * Math.cos(h), sign * Math.sin(h), 0, 0];
  };

  it("midpoint of 0° and 10° roll is 5° even when the second sample is stored negated", () => {
    for (const sign of [1, -1]) {
      const s = new Replay(block(rollQuat(0), rollQuat(10, sign))).at(0.5);
      expect(Math.hypot(...s.quaternion)).toBeCloseTo(1, 12);
      expect(Math.abs(s.eulerDeg[0])).toBeCloseTo(5, 2);
    }
  });
});

describe("wind in the recording (generated by Python with a 3D wind + gusts)", () => {
  it("reproduces wind, air-relative speed and limit flag at every golden query", () => {
    expect(rec.wind?.config.speed_mps).toBe(3);
    for (const q of golden.queries as unknown as { t_query: number; wind_mps: number[]; airspeed_mps: number; limited: boolean }[]) {
      const s = replay.at(q.t_query);
      close(s.wind, q.wind_mps);
      close(s.airspeed, q.airspeed_mps, 1e-9);
      expect(s.limited).toBe(q.limited);
    }
    const w = replay.at(2).wind;
    expect(Math.hypot(...w)).toBeGreaterThan(1); // a real, non-zero wind
  });
  it("older recordings without wind columns still replay (zero wind, airspeed = ground speed)", () => {
    const old = { ...flight } as FlightBlock;
    delete old.wind_mps;
    delete old.limited;
    const s = new Replay(old).at(1.0);
    expect(s.wind).toEqual([0, 0, 0]);
    expect(s.limited).toBe(false);
    expect(s.airspeed).toBeCloseTo(Math.hypot(...s.velocity), 12);
  });
});
