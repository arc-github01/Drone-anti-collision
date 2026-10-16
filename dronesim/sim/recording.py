"""Recording and replay of a simulation run.

`build_recording` turns a finished run into a JSON-serialisable dict (schema sim_recording/v1);
`Replay` reads it back and answers "what was the state at time t?" using only NumPy - no YOLO,
OpenCV, planner or physics - so graphs and 3D playback never need the run to be recomputed.

Deterministic by design: no wall-clock timestamps, so identical runs give identical files.
"""
import json
from dataclasses import asdict
from typing import Optional

import numpy as np

from dronesim.sim import quaternion as quat
from dronesim.sim.wind import components_in_approach_frame

SCHEMA = "sim_recording/v1"
_DECIMALS = 6


def _a(x) -> list:
    return np.round(np.asarray(x, dtype=float), _DECIMALS).tolist()


def _flight_block(hist: dict, env) -> dict:
    q = hist["quaternion"]
    euler = np.degrees([quat.to_euler(qi) for qi in q])
    return {
        "t": _a(hist["t"]),
        "position_m": _a(hist["position"]),
        "velocity_mps": _a(hist["velocity"]),
        "quaternion_wxyz": _a(q),
        "euler_rpy_deg": _a(euler),
        "body_rates_radps": _a(hist["body_rates"]),
        "motor_command": _a(hist["command"]),        # (N,4) controller outputs in [0,1]  <- the graphs
        "motor_omega_radps": _a(hist["omega"]),      # (N,4)
        "motor_thrust_n": _a(hist["thrust"]),        # (N,4)
        "motor_power_w": _a(hist["power_w"]),        # (N,4) electrical
        "setpoint_position_m": _a(hist["setpoint_position"]),
        "clearance_m": _a(env.clearance(hist["position"])),
        "saturated": [bool(v) for v in hist["saturated"]],
        "limited": [bool(v) for v in hist["limited"]],               # tilt/acceleration limit clipped the demand
        "wind_mps": _a(hist["wind"]),                                # wind velocity the drag law used (world frame)
        "disturbance_est_mps2": _a(hist["disturbance_est"]),         # controller's observer estimate of drag accel
    }


def _traj_block(traj, stride: int) -> dict:
    return {"t": _a(traj.t[::stride]), "position_m": _a(traj.pos[::stride]),
            "velocity_mps": _a(traj.vel[::stride])}


def _summary(s) -> Optional[dict]:
    return None if s is None else asdict(s)


def build_recording(run, params, cfg, source: Optional[dict] = None) -> dict:
    """`run` is a runner.RunResult; `params` DroneParams; `cfg` the RunConfig used."""
    env, pr = run.environment, run.plan_result
    box = env.boxes[0]
    stride = max(1, int(round(0.05 / cfg.planner.dt_s)))
    sc = run.scenario
    scale = None
    if cfg.approach_speed_mps is None or abs(cfg.approach_speed_mps - sc["approach_speed_mps"]) < 1e-9:
        scale = 1.0
    elif cfg.approach_speed_mps > 0:
        scale = sc["approach_speed_mps"] / cfg.approach_speed_mps   # video seconds per sim second
    plan = None
    if pr.plan is not None:
        plan = {"kind": pr.plan.kind, "params": pr.plan.params, "metrics": pr.plan.metrics,
                "trajectory": _traj_block(pr.plan.trajectory, stride)}
    return {
        "schema": SCHEMA,
        "source": source or {},
        "physics": {"dt_s": cfg.physics_dt_s, "record_every": cfg.record_every,
                    "sample_dt_s": cfg.physics_dt_s * cfg.record_every, "control_hz": cfg.control_hz},
        "frames": {"world": "ENU (x forward at start, y left, z up)", "body": "FLU",
                   "quaternion": "[w,x,y,z] body->world", "units": "SI"},
        "drone": {"params": asdict(params), "k_thrust": params.k_thrust, "k_torque": params.k_torque,
                  "hover_command": params.hover_command, "radius_m": env.drone_radius_m,
                  "motor_layout": {"positions_xy_m": params.motor_positions.tolist(),
                                   "spin_dir": params.spin_dir.tolist(),
                                   "names": ["M1 front-right CCW", "M2 rear-right CW",
                                             "M3 rear-left CCW", "M4 front-left CW"]}},
        "environment": {"obstacles": [{"center_m": box.center.tolist(), "half_extents_m": box.half.tolist()}],
                        "safety_margin_m": env.safety_margin_m, "floor_z_m": env.floor_z_m},
        "scenario": sc,
        "start_m": _a(run.start),
        "approach_speed_mps": run.approach_speed_mps,
        "video_sync": {"reference_video_time_s": sc.get("reference_time_s", 0.0), "time_scale": scale,
                       "note": "video_t = reference + sim_t * time_scale; None/approximate if speed was overridden"},
        "assumptions": ["Drone parameters, drone radius, safety margin, start altitude and floor are ASSUMED values.",
                        "Obstacle metric size comes from configuration, not measurement: "
                        f"size_is_assumed={sc.get('assumptions', {}).get('size_is_assumed')}, "
                        f"fov_is_assumed={sc.get('assumptions', {}).get('fov_is_assumed')}.",
                        "Controller reads the true simulated state (no sensor noise). Wind: none."],
        "wind": _wind_block(run, params, cfg),
        "planning": {"needs_avoidance": pr.needs_avoidance, "note": pr.note, "n_candidates": pr.n_candidates,
                     "n_feasible": pr.n_feasible, "rejections": pr.rejections, "nominal_check": pr.nominal_check,
                     "nominal_limits": pr.nominal_limits, "nominal_violations": list(pr.nominal_violations),
                     "attempts": run.replans,
                     "plan": plan, "nominal_trajectory": _traj_block(pr.nominal, stride)},
        "flight": None if run.history is None else _flight_block(run.history, env),
        "summary": _summary(run.summary),
        "nominal_flight": None if run.nominal_history is None else _flight_block(run.nominal_history, env),
        "nominal_summary": _summary(run.nominal_summary),
    }


def _wind_block(run, params, cfg) -> dict:
    """Wind description + what it demands of the drone, for display and for reproducing the run."""
    from dronesim.sim.planner import station_keeping_requirement
    w = run.wind
    mean = w.mean_vector()
    return {
        "config": w.to_dict(),
        "mean_vector_mps": _a(mean),
        "components_approach_frame": components_in_approach_frame(mean),
        "station_keeping": station_keeping_requirement(params, mean),
        "preroll_s": cfg.preroll_s if w.active else 0.0,
        "convention": "direction the wind blows TOWARD; azimuth from +x (forward) toward +y (left) in the horizontal "
                      "plane, elevation from the horizontal plane toward +z (up); "
                      "w = V (cos(el)cos(az), cos(el)sin(az), sin(el))",
        "model": "steady wind + per-axis Ornstein-Uhlenbeck gusts (exact discretisation, seeded); drag "
                 "F = -c_d |v - w| (v - w)",
    }


def save_recording(rec: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))


def load_recording(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported recording schema: {rec.get('schema')!r} (expected {SCHEMA})")
    return rec


class Replay:
    """Random-access playback of one flight block ('flight' or 'nominal_flight').

    `at(t)` is a pure function of t: scrubbing in any order gives identical results.
    Positions/rates interpolate linearly, attitude by normalised lerp, and motor
    commands are zero-order-held (that is how the controller applied them).
    """

    def __init__(self, rec: dict, which: str = "flight"):
        blk = rec.get(which)
        if blk is None:
            raise ValueError(f"Recording has no '{which}' data")
        self.rec, self.blk = rec, blk
        self.t = np.asarray(blk["t"], dtype=float)
        self._arr = {k: np.asarray(v, dtype=float) for k, v in blk.items() if k not in ("t", "saturated", "limited")}
        self._sat = np.asarray(blk["saturated"], dtype=bool)
        self._lim = np.asarray(blk.get("limited", [False] * len(self.t)), dtype=bool)   # absent in old recordings
        zeros = np.zeros((len(self.t), 3))
        self._arr.setdefault("wind_mps", zeros)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def index_at(self, t: float) -> int:
        """Last sample with time <= t (clamped)."""
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def at(self, t: float) -> dict:
        t = float(np.clip(t, self.t[0], self.t[-1]))
        i = self.index_at(t)
        j = min(i + 1, len(self.t) - 1)
        a = 0.0 if j == i else (t - self.t[i]) / (self.t[j] - self.t[i])
        lerp = lambda k: (1 - a) * self._arr[k][i] + a * self._arr[k][j]
        q0, q1 = self._arr["quaternion_wxyz"][i], self._arr["quaternion_wxyz"][j]
        if q0 @ q1 < 0:
            q1 = -q1
        q = quat.normalize((1 - a) * q0 + a * q1)
        hold = lambda k: self._arr[k][i]
        return {
            "t": t, "index": i,
            "position_m": lerp("position_m"), "velocity_mps": lerp("velocity_mps"),
            "quaternion_wxyz": q, "euler_rpy_deg": np.degrees(quat.to_euler(q)),
            "body_rates_radps": lerp("body_rates_radps"),
            "motor_command": hold("motor_command"), "motor_omega_radps": lerp("motor_omega_radps"),
            "motor_thrust_n": lerp("motor_thrust_n"), "motor_power_w": lerp("motor_power_w"),
            "setpoint_position_m": lerp("setpoint_position_m"), "clearance_m": float(lerp("clearance_m")),
            "saturated": bool(self._sat[i]), "limited": bool(self._lim[i]),
            "wind_mps": lerp("wind_mps"),
            "airspeed_mps": float(np.linalg.norm(lerp("velocity_mps") - lerp("wind_mps"))),   # |v - w|
        }

    def motor_series(self, motor: int, quantity: str = "motor_command"):
        """(t, values) for one motor (0..3): what each of the four graphs plots."""
        return self.t, self._arr[quantity][:, motor]

    def path(self, upto_t: Optional[float] = None) -> np.ndarray:
        """Flown positions up to time t (for drawing the actual trajectory trail)."""
        n = len(self.t) if upto_t is None else self.index_at(upto_t) + 1
        return self._arr["position_m"][:n]

    def video_time(self, t: float) -> Optional[float]:
        s = self.rec["video_sync"]
        return None if s["time_scale"] is None else s["reference_video_time_s"] + t * s["time_scale"]

    def recompute_clearance(self) -> np.ndarray:
        """Clearance recomputed from the recorded geometry alone (consistency check)."""
        env, r = self.rec["environment"], self.rec["drone"]["radius_m"]
        pts = self._arr["position_m"]
        d = []
        for b in env["obstacles"]:
            q = np.abs(pts - np.asarray(b["center_m"])) - np.asarray(b["half_extents_m"])
            d.append(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0))
        c = np.min(d, axis=0) - r
        if env["floor_z_m"] is not None:
            c = np.minimum(c, pts[:, 2] - env["floor_z_m"] - r)
        return c
