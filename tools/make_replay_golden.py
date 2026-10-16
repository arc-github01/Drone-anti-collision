"""Generate a small recording plus the answers Python's Replay gives for it, so the TypeScript
Replay (frontend/src/replay.ts) can be tested against the reference implementation.

Usage: python tools/make_replay_golden.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dronesim.sim.params import DroneParams  # noqa: E402
from dronesim.sim.recording import Replay, build_recording  # noqa: E402
from dronesim.sim.runner import RunConfig, run_avoidance  # noqa: E402
from dronesim.sim.wind import WindConfig  # noqa: E402

OUT = os.path.join(ROOT, "frontend", "src", "test", "fixtures", "replay_golden.json")

scenario = {"approach_speed_mps": 1.5, "reference_time_s": 0.5,
            "drone": {"position_m": [0, 0, 0], "velocity_mps": [1.5, 0, 0]},
            "obstacle": {"center_m": [2.7, 0.0, 0.0], "width_m": 0.5, "height_m": 0.5,
                         "thickness_m": 0.05, "normal": [-1, 0, 0], "static": True},
            "assumptions": {"size_is_assumed": True, "fov_is_assumed": True}}
cfg = RunConfig(record_every=50, fly_without_avoidance=False, wind=WindConfig(3, 60, 20, 0.8, 2.0, 5))   # 3D wind + gusts, coarse 0.1 s samples
params = DroneParams()
run = run_avoidance(scenario, params, cfg)
rec = build_recording(run, params, cfg)
rp = Replay(rec)

rng = np.random.default_rng(7)
times = sorted(set([-3.0, 0.0, 0.05, rp.duration, rp.duration + 5.0] + rng.uniform(0, rp.duration, 40).round(4).tolist()
                   + [float(rp.t[3]), float(rp.t[10]) + 1e-9]))
keys = ["position_m", "velocity_mps", "quaternion_wxyz", "euler_rpy_deg", "body_rates_radps", "motor_command",
        "motor_omega_radps", "motor_thrust_n", "motor_power_w", "setpoint_position_m", "wind_mps"]
queries = []
for t in times:
    s = rp.at(t)
    q = {"t_query": t, "t": s["t"], "index": s["index"], "clearance_m": s["clearance_m"], "saturated": s["saturated"], "limited": s["limited"], "airspeed_mps": s["airspeed_mps"]}
    q.update({k: np.asarray(s[k]).tolist() for k in keys})
    queries.append(q)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"recording": rec, "queries": queries}, f, separators=(",", ":"))
print(f"wrote {OUT}: {len(rec['flight']['t'])} samples, {len(queries)} queries, {os.path.getsize(OUT) / 1e3:.0f} kB")
