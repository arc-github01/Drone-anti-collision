"""End-to-end run: scenario -> prediction -> plan -> closed-loop flight (controller + physics).

The planner proposes a trajectory; the flight controller turns it into motor commands; the
physics engine decides what actually happens. Reported clearances are measured on the ACTUAL
flown path, not on the plan.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dronesim.sim import quaternion as quat
from dronesim.sim.collision import Box, Environment, predict_collision
from dronesim.sim.controller import FlightController, Gains, Setpoint
from dronesim.sim.loop import simulate
from dronesim.sim.params import DroneParams
from dronesim.sim.planner import PlannerConfig, PlanResult, Trajectory, plan_avoidance
from dronesim.sim.quadrotor import Quadrotor
from dronesim.sim.wind import WindConfig, WindModel


@dataclass
class RunConfig:
    start_altitude_m: float = 1.5          # assumed drone height above the floor (world z = 0)
    floor_z_m: Optional[float] = 0.0
    drone_radius_m: float = 0.10
    safety_margin_m: float = 0.25
    approach_speed_mps: Optional[float] = None   # override the video-derived speed (e.g. a faster demo)
    physics_dt_s: float = 0.002
    control_hz: float = 250.0
    record_every: int = 2                  # history stored at physics_dt * record_every
    tail_s: float = 1.0                    # extra flight time after the plan ends
    fly_without_avoidance: bool = True     # also fly the straight path to show what would have happened
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    wind: WindConfig = field(default_factory=WindConfig)   # steady 3D wind (+ optional seeded gusts); default: still air
    max_replans: int = 2                   # closed-loop re-plans if gusts ate the clearance margin
    preroll_s: float = 6.0                 # unrecorded straight flight before t = 0 when wind is on, so the drone
                                           # (and its disturbance observer) start in equilibrium with the wind


@dataclass
class FlightSummary:
    min_clearance_m: float
    t_min_clearance_s: float
    collided: bool
    max_tilt_deg: float
    peak_speed_mps: float
    max_tracking_error_m: float
    saturated_fraction: float
    limited_fraction: float = 0.0     # share of the flight with the tilt/acceleration limit clipping the demand


@dataclass
class RunResult:
    scenario: dict
    environment: Environment
    start: np.ndarray
    approach_speed_mps: float
    plan_result: PlanResult
    history: Optional[dict]               # actual flight following the plan (None if unavoidable and no flight)
    summary: Optional[FlightSummary]
    nominal_history: Optional[dict]       # actual flight of the straight path, no avoidance
    nominal_summary: Optional[FlightSummary]
    wind: WindConfig = field(default_factory=WindConfig)
    replans: list = field(default_factory=list)   # [{"extra_buffer_m", "min_clearance_m"}] of each flown attempt


def environment_from_scenario(scenario: dict, cfg: RunConfig) -> Environment:
    box = Box.from_obstacle(scenario["obstacle"], z_offset=cfg.start_altitude_m)
    return Environment([box], cfg.drone_radius_m, cfg.safety_margin_m, cfg.floor_z_m)


def _tilt_deg(quats: np.ndarray) -> np.ndarray:
    z = np.array([quat.to_rotation_matrix(q)[2, 2] for q in quats])
    return np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))


def _summarise(hist: dict, env: Environment) -> FlightSummary:
    c = env.clearance(hist["position"])
    i = int(np.argmin(c))
    err = np.linalg.norm(hist["position"] - hist["setpoint_position"], axis=1)
    return FlightSummary(float(c[i]), float(hist["t"][i]), bool(c[i] < 0.0),
                         float(_tilt_deg(hist["quaternion"]).max()),
                         float(np.linalg.norm(hist["velocity"], axis=1).max()),
                         float(err.max()), float(np.mean(hist["saturated"])),
                         float(np.mean(hist["limited"])))


def fly_trajectory(traj: Trajectory, start, v0, params: DroneParams, cfg: RunConfig, duration_s: float,
                   gains: Optional[Gains] = None) -> dict:
    wind_fn = WindModel(cfg.wind, duration_s + 1.0) if cfg.wind.active else None   # same seed => same gusts every flight
    quad = Quadrotor(params, dt=cfg.physics_dt_s, wind_fn=wind_fn)
    ctrl = FlightController(params, gains)
    start, v0 = np.asarray(start, dtype=float), np.asarray(v0, dtype=float)
    pre = cfg.preroll_s if cfg.wind.active else 0.0
    if pre > 0:
        # Fly the straight approach line for `pre` seconds at NEGATIVE time (the gust series is held at its t = 0
        # value there), arriving at `start` with velocity v0 and the controller settled; this history is discarded.
        quad.reset(position=start - v0 * pre, velocity=v0)
        quad.n_steps = -int(round(pre / quad.dt))
        simulate(quad, ctrl, lambda t: Setpoint(start + v0 * t, v0), pre, cfg.control_hz, 10**9)
        assert quad.n_steps == 0
    else:
        quad.reset(position=start, velocity=v0)

    def sp(t):
        p, v, a = traj.at(t)
        return Setpoint(p, v, a, yaw=0.0)

    return simulate(quad, ctrl, sp, duration_s, cfg.control_hz, cfg.record_every)


def run_avoidance(scenario: dict, params: Optional[DroneParams] = None,
                  cfg: Optional[RunConfig] = None) -> RunResult:
    params, cfg = params or DroneParams(), cfg or RunConfig()
    env = environment_from_scenario(scenario, cfg)
    v = cfg.approach_speed_mps if cfg.approach_speed_mps is not None else scenario["approach_speed_mps"]
    start = np.array([0.0, 0.0, cfg.start_altitude_m]) + np.array(scenario["drone"]["position_m"])
    v0 = np.array([v, 0.0, 0.0])

    wind_mean = cfg.wind.mean_vector()
    pr = plan_avoidance(env, start, v, params, cfg.planner, wind=wind_mean)
    history = summary = nom_hist = nom_sum = None
    replans: list = []

    if cfg.fly_without_avoidance and pr.needs_avoidance:
        nom_hist = fly_trajectory(pr.nominal, start, v0, params, cfg, pr.nominal.t[-1])
        nom_sum = _summarise(nom_hist, env)

    # Plan -> fly -> verify. Gusts push the drone off the planned path; if the FLOWN clearance fell below the
    # safety margin, plan again demanding more clearance (only meaningful with turbulence).
    extra = 0.0
    while pr.plan is not None:
        chosen = pr.plan.trajectory
        history = fly_trajectory(chosen, start, v0, params, cfg, chosen.t[-1] + cfg.tail_s)
        summary = _summarise(history, env)
        replans.append({"extra_buffer_m": extra, "min_clearance_m": summary.min_clearance_m})
        if summary.min_clearance_m >= env.safety_margin_m or len(replans) > cfg.max_replans or not cfg.wind.active:
            break
        extra += (env.safety_margin_m - summary.min_clearance_m) + 0.05
        nxt = plan_avoidance(env, start, v, params, cfg.planner, wind=wind_mean, extra_buffer_m=extra)
        if nxt.plan is None:          # cannot add more clearance within the limits: keep the verified flight
            break
        pr = nxt
    if pr.plan is None and not pr.needs_avoidance:
        history = nom_hist = fly_trajectory(pr.nominal, start, v0, params, cfg, pr.nominal.t[-1])
        summary = nom_sum = _summarise(history, env)
    return RunResult(scenario, env, start, v, pr, history, summary, nom_hist, nom_sum, cfg.wind, replans)
