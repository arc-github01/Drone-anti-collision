"""Collision prediction and avoidance trajectory selection.

Candidates are smooth (minimum-jerk) sidesteps from a straight forward cruise: offset by D in one
of 8 directions, hold while passing the obstacle, then return to the original line. Each candidate
is checked against the drone's limits (acceleration, thrust, tilt) and against obstacle/floor
clearance; infeasible ones are REJECTED with reasons recorded. If no detour is feasible a braking
stop is tried; if that also fails the plan is None (collision unavoidable at these limits).

Assumes the cruise heading is +x (the camera/approach axis in the scenario).

Wind: the planner works in the inertial (ground) frame, where the path kinematics are unchanged by wind; what wind
changes is the force the rotors must supply. With drag a_D = -(c_d/m)|v - w|(v - w) the rotors must provide

    a_req = a_plan - a_D = a_plan + (c_d/m)|v - w|(v - w),     specific force f = a_req + g e_z,

and the acceleration / thrust / tilt limits are checked on a_req and f (not on a_plan). The steady mean wind is
used; gust-induced tracking error is handled by the runner's closed-loop verification and re-planning.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dronesim.sim.collision import Environment, predict_collision
from dronesim.sim.params import GRAVITY, DroneParams


@dataclass
class PlannerConfig:
    dt_s: float = 0.02
    limit_fraction: float = 0.8          # use at most this fraction of the accel/thrust/tilt limits
    tracking_buffer_m: float = 0.05      # extra planned clearance, reserved for path-tracking error
    offsets_m: tuple = tuple(np.round(np.arange(0.2, 2.01, 0.1), 2))
    direction_deg: tuple = (0, 45, 90, 135, 180, 225, 270, 315)   # 0 = +y (left), 90 = up
    blend_s: tuple = (1.0, 2.0, 3.0)
    rise_end_fractions: tuple = (0.6, 0.9)   # offset reached at this fraction of time-to-front
    vertical_penalty_m: float = 0.15     # prefer sideways dodges (they keep altitude and use differential thrust)
    accel_cost: float = 0.1
    horizon_s: float = 40.0


@dataclass
class Trajectory:
    t: np.ndarray
    pos: np.ndarray    # (N,3)
    vel: np.ndarray
    acc: np.ndarray

    def at(self, t: float):
        """(pos, vel, acc) at time t; linear interpolation, constant-velocity extrapolation after the end."""
        if t >= self.t[-1]:
            dt = t - self.t[-1]
            return self.pos[-1] + self.vel[-1] * dt, self.vel[-1].copy(), np.zeros(3)
        f = lambda a: np.array([np.interp(t, self.t, a[:, k]) for k in range(3)])
        return f(self.pos), f(self.vel), f(self.acc)


@dataclass
class Plan:
    kind: str                      # "detour" | "brake"
    params: dict
    trajectory: Trajectory
    metrics: dict


@dataclass
class PlanResult:
    needs_avoidance: bool
    nominal: Trajectory
    nominal_check: dict
    plan: Optional[Plan]
    n_candidates: int
    n_feasible: int
    rejections: dict               # reason -> count
    note: str = ""
    wind_mps: tuple = (0.0, 0.0, 0.0)
    nominal_limits: Optional[dict] = None       # what the straight path demands of the rotors (in this wind)
    nominal_violations: tuple = ()               # which limits the straight path alone already exceeds


def _minjerk(tau):
    """Smooth 0->1 step and its derivatives w.r.t. tau in [0,1]."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def _nominal(start, v, t):
    pos = np.asarray(start, dtype=float) + np.outer(t, [v, 0.0, 0.0])
    vel = np.tile([v, 0.0, 0.0], (len(t), 1))
    return Trajectory(t, pos, vel, np.zeros_like(pos))


def _detour(start, v, t, direction_deg, D, Tb, t1, t2) -> Trajectory:
    base = _nominal(start, v, t)
    u = np.array([0.0, np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    sr, dsr, ddsr = _minjerk((t - t1) / Tb)
    sf, dsf, ddsf = _minjerk((t - t2) / Tb)
    s, ds, dds = sr - sf, (dsr - dsf) / Tb, (ddsr - ddsf) / Tb**2
    return Trajectory(t, base.pos + D * np.outer(s, u), base.vel + D * np.outer(ds, u),
                      base.acc + D * np.outer(dds, u))


def _brake(start, v, t, t_stop, T) -> Trajectory:
    sg, dsg, ddsg = _minjerk(t / T)          # speed fraction removed
    tau = np.clip(t / T, 0.0, 1.0)
    # x(t) = v * (t_c - T * integral_0^tau sigma),  integral of the min-jerk step = 2.5 tau^4 - 3 tau^5 + tau^6
    x = v * (T * tau - T * (2.5 * tau**4 - 3 * tau**5 + tau**6))
    pos = np.asarray(start, dtype=float) + np.outer(x, [1.0, 0.0, 0.0])
    vel = np.outer(v * (1 - sg), [1.0, 0.0, 0.0])
    acc = np.outer(-v * dsg / T, [1.0, 0.0, 0.0])
    return Trajectory(t, pos, vel, acc)


def required_acceleration(traj: Trajectory, p: DroneParams, wind=None) -> np.ndarray:
    """Acceleration the ROTORS must supply (per sample): a_plan + (c_d/m)|v - w|(v - w)."""
    w = np.zeros(3) if wind is None else np.asarray(wind, dtype=float)
    v_rel = traj.vel - w
    k = p.drag_coeff_kg_per_m / p.mass_kg
    return traj.acc + k * np.linalg.norm(v_rel, axis=1, keepdims=True) * v_rel


def station_keeping_requirement(p: DroneParams, wind) -> dict:
    """What it takes to hover in a steady wind: thrust/tilt/acceleration needed at v = 0.

    At v = 0 the air-relative velocity is -w, so drag accelerates the drone DOWNWIND: a_D = +(c_d/m)|w| w.
    The rotors must cancel it:  a_req = -a_D,  specific force f = a_req + g e_z  (lean INTO the wind;
    an updraft unloads the rotors, a downdraft loads them)."""
    w = np.asarray(wind, dtype=float)
    a = -(p.drag_coeff_kg_per_m / p.mass_kg) * np.linalg.norm(w) * w
    f = a + np.array([0.0, 0.0, GRAVITY])
    return {"accel_mps2": float(np.linalg.norm(a)), "thrust_n": float(p.mass_kg * np.linalg.norm(f)),
            "tilt_deg": float(np.degrees(np.arctan2(np.linalg.norm(f[:2]), f[2])))}


def trajectory_limits(traj: Trajectory, p: DroneParams, wind=None) -> dict:
    """Thrust/tilt/acceleration the PLAN demands of the rotors (assuming it is tracked perfectly), in a steady wind."""
    a_req = required_acceleration(traj, p, wind)
    f = a_req + np.array([0.0, 0.0, GRAVITY])             # specific force
    fn = np.linalg.norm(f, axis=1)
    tilt = np.degrees(np.arccos(np.clip(f[:, 2] / fn, -1, 1)))
    return {"peak_accel_mps2": float(np.linalg.norm(a_req, axis=1).max()),
            "peak_thrust_n": float(p.mass_kg * fn.max()),
            "min_thrust_n": float(p.mass_kg * fn.min()),
            "peak_tilt_deg": float(tilt.max())}


def _violations(lim: dict, p: DroneParams, frac: float) -> list:
    out = []
    if lim["peak_accel_mps2"] > frac * p.max_accel_mps2:
        out.append("accel")
    if lim["peak_thrust_n"] > frac * 4 * p.max_thrust_per_motor_n:
        out.append("thrust")
    if lim["peak_tilt_deg"] > frac * p.max_tilt_deg:
        out.append("tilt")
    return out


def plan_avoidance(env: Environment, start, approach_speed_mps: float, params: DroneParams,
                   cfg: Optional[PlannerConfig] = None, wind=None, extra_buffer_m: float = 0.0) -> PlanResult:
    """`wind`: steady world-frame wind vector (m/s). `extra_buffer_m`: additional planned clearance (used by the
    runner's closed-loop re-planning when gusts caused too much tracking error)."""
    cfg = cfg or PlannerConfig()
    start = np.asarray(start, dtype=float)
    wind = np.zeros(3) if wind is None else np.asarray(wind, dtype=float)
    wind_t = tuple(float(x) for x in wind)
    v = float(approach_speed_mps)
    box = env.boxes[0]
    x_front = box.center[0] - box.half[0] - env.drone_radius_m - env.safety_margin_m
    x_back = box.center[0] + box.half[0] + env.drone_radius_m + env.safety_margin_m
    t_front = max((x_front - start[0]) / v, 0.0)
    t_back = (x_back - start[0]) / v

    # Nominal (intended) straight path and its collision prediction.
    t_nom = np.arange(0.0, min(t_back + 2.0, cfg.horizon_s), cfg.dt_s)
    nominal = _nominal(start, v, t_nom)
    nominal_check = predict_collision(env, t_nom, nominal.pos)
    nom_lim = trajectory_limits(nominal, params, wind)
    nom_viol = tuple(_violations(nom_lim, params, cfg.limit_fraction))
    wind_txt = (f" Wind {np.linalg.norm(wind):.1f} m/s: the straight path alone needs {nom_lim['peak_tilt_deg']:.0f} deg tilt "
                f"/ {nom_lim['peak_thrust_n']:.2f} N and exceeds the {'/'.join(nom_viol)} limit.") if nom_viol else ""
    if not nominal_check["violates_margin"]:
        note = "nominal path is clear" + (" (but the wind exceeds the drone's limits." + wind_txt + ")" if nom_viol else "")
        return PlanResult(False, nominal, nominal_check, None, 0, 0, {}, note, wind_t, nom_lim, nom_viol)

    need = env.safety_margin_m + cfg.tracking_buffer_m + extra_buffer_m   # planned clearance required of any candidate
    rejections = {"accel": 0, "thrust": 0, "tilt": 0, "clearance": 0, "floor": 0}
    best, best_cost, n, n_ok = None, np.inf, 0, 0
    for phi in cfg.direction_deg:
        for D in cfg.offsets_m:
            for Tb in cfg.blend_s:
                for e in cfg.rise_end_fractions:
                    t1 = max(0.0, e * t_front - Tb)
                    t2 = t_back + 0.3
                    t = np.arange(0.0, t2 + Tb + 1.0, cfg.dt_s)
                    traj = _detour(start, v, t, phi, D, Tb, t1, t2)
                    n += 1
                    lim = trajectory_limits(traj, params, wind)
                    why = _violations(lim, params, cfg.limit_fraction)
                    c_obs = env.obstacle_clearance(traj.pos).min()
                    if c_obs < need:
                        why.append("clearance")
                    if env.floor_clearance(traj.pos).min() < need:
                        why.append("floor")
                    for w in why:
                        rejections[w] += 1
                    if why:
                        continue
                    n_ok += 1
                    cost = D + cfg.accel_cost * lim["peak_accel_mps2"] \
                        + (cfg.vertical_penalty_m if phi in (90, 270) else 0.0)
                    if cost < best_cost:
                        chk = predict_collision(env, t, traj.pos)
                        best_cost = cost
                        best = Plan("detour", dict(direction_deg=float(phi), offset_m=float(D), blend_s=float(Tb),
                                                   rise_start_s=float(t1), return_start_s=float(t2), cost=float(cost)),
                                    traj, {**lim, "min_clearance_m": chk["min_clearance_m"]})
    if best is not None:
        return PlanResult(True, nominal, nominal_check, best, n, n_ok, rejections, "detour selected",
                          wind_t, nom_lim, nom_viol)

    # Fallback: braking stop before the obstacle.
    d_avail = x_front - start[0]
    if d_avail > 0:
        for scale in (1.0, 0.9, 0.8):
            T = 2.0 * d_avail * scale / v
            t = np.arange(0.0, T + 2.0, cfg.dt_s)
            traj = _brake(start, v, t, T, T)
            lim = trajectory_limits(traj, params, wind)
            if not _violations(lim, params, cfg.limit_fraction) and \
                    env.obstacle_clearance(traj.pos).min() >= need:
                chk = predict_collision(env, t, traj.pos)
                return PlanResult(True, nominal, nominal_check,
                                  Plan("brake", dict(stop_time_s=float(T)), traj,
                                       {**lim, "min_clearance_m": chk["min_clearance_m"]}),
                                  n, n_ok, rejections, "no feasible detour; braking stop selected",
                                  wind_t, nom_lim, nom_viol)
    return PlanResult(True, nominal, nominal_check, None, n, n_ok, rejections,
                      "UNAVOIDABLE within limits: no detour or braking stop is feasible." + wind_txt,
                      wind_t, nom_lim, nom_viol)
