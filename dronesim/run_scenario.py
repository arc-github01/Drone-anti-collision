"""Run the avoidance simulation for a processed video and write sim_recording.json.

Usage: python -m dronesim.run_scenario data/runs/testv2/obstacle_metric.json [--speed M_PER_S]
         [--altitude M] [--floor M | --no-floor] [--margin M] [--dodge left|right|up|down|auto]
         [--wind-speed V --wind-azimuth DEG --wind-elevation DEG --turbulence SIGMA --turbulence-tau S --wind-seed N]
         [--out PATH]
"""
import argparse
import json
import os

from dronesim.sim.params import DroneParams
from dronesim.sim.recording import build_recording, save_recording
from dronesim.sim.runner import RunConfig, run_avoidance
from dronesim.sim.wind import WindConfig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("metric_json", help="obstacle_metric.json from dronesim.vision.reconstruction")
    ap.add_argument("--speed", type=float, default=None, help="override approach speed (m/s)")
    ap.add_argument("--altitude", type=float, default=None, help="drone start altitude above floor (m)")
    ap.add_argument("--floor", type=float, default=None, help="floor height (m)")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--margin", type=float, default=None, help="safety margin (m)")
    ap.add_argument("--dodge", choices=["auto", "left", "right", "up", "down"], default="auto",
                    help="restrict the detour direction (default: planner chooses, preferring sideways)")
    ap.add_argument("--wind-speed", type=float, default=0.0, help="wind speed (m/s)")
    ap.add_argument("--wind-azimuth", type=float, default=0.0,
                    help="direction the wind blows TOWARD in the horizontal plane, deg from +x (forward) toward +y (left)")
    ap.add_argument("--wind-elevation", type=float, default=0.0, help="deg above the horizontal (+90 = updraft)")
    ap.add_argument("--turbulence", type=float, default=0.0, help="per-axis gust standard deviation (m/s)")
    ap.add_argument("--turbulence-tau", type=float, default=2.0, help="gust correlation time (s)")
    ap.add_argument("--wind-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    with open(a.metric_json, encoding="utf-8") as f:
        metric = json.load(f)
    if metric.get("scenario") is None:
        raise SystemExit("No scenario in this file (no measurable approach was detected).")
    cfg, params = RunConfig(), DroneParams()
    if a.speed is not None: cfg.approach_speed_mps = a.speed
    if a.altitude is not None: cfg.start_altitude_m = a.altitude
    if a.floor is not None: cfg.floor_z_m = a.floor
    if a.no_floor: cfg.floor_z_m = None
    if a.margin is not None: cfg.safety_margin_m = a.margin

    if a.dodge != "auto":
        cfg.planner.direction_deg = ({"left": (0,), "up": (90,), "right": (180,), "down": (270,)}[a.dodge])

    cfg.wind = WindConfig(a.wind_speed, a.wind_azimuth, a.wind_elevation, a.turbulence, a.turbulence_tau, a.wind_seed)

    run = run_avoidance(metric["scenario"], params, cfg)
    rec = build_recording(run, params, cfg, source={"metric_json": os.path.basename(a.metric_json)})
    out = a.out or os.path.join(os.path.dirname(a.metric_json), "sim_recording.json")
    save_recording(rec, out)

    pr = run.plan_result
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    print(f"planning: {pr.note}")
    if run.summary:
        print(f"flown min clearance {run.summary.min_clearance_m:.3f} m, collided={run.summary.collided}")
    if run.nominal_summary:
        print(f"without avoidance: min clearance {run.nominal_summary.min_clearance_m:.3f} m, "
              f"collided={run.nominal_summary.collided}")


if __name__ == "__main__":
    main()
