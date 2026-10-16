// Random-access playback of one recorded flight. A TypeScript port of dronesim/sim/recording.py `Replay`:
// at(t) is a pure function of t (scrub in any order, same answer); positions/rates interpolate linearly,
// attitude by normalised lerp, motor commands are zero-order-held (that is how the controller applied them).
import type { FlightBlock, MotorQuantity, Vec3, Vec4 } from "./types";

export interface ReplayState {
  t: number;
  index: number;
  position: Vec3;
  velocity: Vec3;
  quaternion: Vec4; // [w,x,y,z] body->world (simulator frame)
  eulerDeg: Vec3; // roll, pitch, yaw
  bodyRates: Vec3;
  motorCommand: Vec4;
  motorOmega: Vec4;
  motorThrust: Vec4;
  motorPower: Vec4;
  setpoint: Vec3;
  clearance: number;
  saturated: boolean;
  limited: boolean;
  wind: Vec3; // wind velocity at that instant (zeros for recordings without wind)
  airspeed: number; // |v - wind|
}

const flat = (rows: number[][], k: number) => {
  const out = new Float64Array(rows.length * k);
  rows.forEach((r, i) => out.set(r, i * k));
  return out;
};

export function quatToEulerDeg(q: Vec4): Vec3 {
  const [w, x, y, z] = q;
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.min(1, Math.max(-1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const d = 180 / Math.PI;
  return [roll * d, pitch * d, yaw * d];
}

export class Replay {
  readonly t: Float64Array;
  readonly n: number;
  private pos: Float64Array;
  private vel: Float64Array;
  private quat: Float64Array;
  private rates: Float64Array;
  private cmd: Float64Array;
  private omega: Float64Array;
  private thrust: Float64Array;
  private power: Float64Array;
  private sp: Float64Array;
  private clr: Float64Array;
  private sat: boolean[];
  private lim: boolean[];
  private wind: Float64Array;

  constructor(f: FlightBlock) {
    this.t = Float64Array.from(f.t);
    this.n = this.t.length;
    this.pos = flat(f.position_m, 3);
    this.vel = flat(f.velocity_mps, 3);
    this.quat = flat(f.quaternion_wxyz, 4);
    this.rates = flat(f.body_rates_radps, 3);
    this.cmd = flat(f.motor_command, 4);
    this.omega = flat(f.motor_omega_radps, 4);
    this.thrust = flat(f.motor_thrust_n, 4);
    this.power = flat(f.motor_power_w, 4);
    this.sp = flat(f.setpoint_position_m, 3);
    this.clr = Float64Array.from(f.clearance_m);
    this.sat = f.saturated;
    this.lim = f.limited ?? new Array<boolean>(this.n).fill(false);
    this.wind = f.wind_mps ? flat(f.wind_mps, 3) : new Float64Array(this.n * 3);
  }

  get duration(): number {
    return this.t[this.n - 1];
  }

  /** Last sample with time <= t (clamped to the ends). */
  indexAt(t: number): number {
    let lo = 0;
    let hi = this.n - 1;
    if (t >= this.t[hi]) return hi;
    if (t < this.t[0]) return 0;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.t[mid] <= t) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  at(tq: number): ReplayState {
    const t = Math.min(this.t[this.n - 1], Math.max(this.t[0], tq));
    const i = this.indexAt(t);
    const j = Math.min(i + 1, this.n - 1);
    const a = j === i ? 0 : (t - this.t[i]) / (this.t[j] - this.t[i]);
    const lerp = (arr: Float64Array, k: number): number[] => {
      const out = new Array<number>(k);
      for (let m = 0; m < k; m++) out[m] = (1 - a) * arr[i * k + m] + a * arr[j * k + m];
      return out;
    };
    const hold = (arr: Float64Array, k: number): number[] => Array.from(arr.subarray(i * k, i * k + k));

    const q0 = Array.from(this.quat.subarray(i * 4, i * 4 + 4));
    let q1 = Array.from(this.quat.subarray(j * 4, j * 4 + 4));
    if (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0) q1 = q1.map((v) => -v);
    const qm = q0.map((v, m) => (1 - a) * v + a * q1[m]);
    const norm = Math.hypot(qm[0], qm[1], qm[2], qm[3]);
    const q = qm.map((v) => v / norm) as Vec4;

    return {
      t,
      index: i,
      position: lerp(this.pos, 3) as Vec3,
      velocity: lerp(this.vel, 3) as Vec3,
      quaternion: q,
      eulerDeg: quatToEulerDeg(q),
      bodyRates: lerp(this.rates, 3) as Vec3,
      motorCommand: hold(this.cmd, 4) as Vec4,
      motorOmega: lerp(this.omega, 4) as Vec4,
      motorThrust: lerp(this.thrust, 4) as Vec4,
      motorPower: lerp(this.power, 4) as Vec4,
      setpoint: lerp(this.sp, 3) as Vec3,
      clearance: lerp(this.clr, 1)[0],
      saturated: this.sat[i],
      limited: this.lim[i],
      wind: lerp(this.wind, 3) as Vec3,
      airspeed: Math.hypot(...lerp(this.vel, 3).map((v, k) => v - lerp(this.wind, 3)[k])),
    };
  }

  /** Values of one motor (0..3) for a quantity: exactly what a graph plots. */
  motorSeries(motor: number, quantity: MotorQuantity = "motor_command"): Float64Array {
    const src = { motor_command: this.cmd, motor_omega_radps: this.omega, motor_thrust_n: this.thrust, motor_power_w: this.power }[quantity];
    const out = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) out[i] = src[i * 4 + motor];
    return out;
  }

  /** Flown positions up to time t (inclusive sample), as a flat xyz array copy. */
  pathUpTo(t: number): Float64Array {
    return this.pos.slice(0, (this.indexAt(t) + 1) * 3);
  }

  get allPositions(): Float64Array {
    return this.pos;
  }
}

/** Video time for a simulation time, or null when the recording has no sync. */
export function videoTime(sync: { reference_video_time_s: number; time_scale: number | null }, t: number): number | null {
  return sync.time_scale === null ? null : sync.reference_video_time_s + t * sync.time_scale;
}
