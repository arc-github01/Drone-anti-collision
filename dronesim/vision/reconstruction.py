"""Pixels -> metric obstacle states and a simulation scenario (pinhole camera model).

Assumptions (all configurable, none measured):
  * obstacle real width/height are supplied (ObstacleConfig); pixels alone cannot give metres;
  * camera looks along the drone's forward axis, level, no lens distortion, hfov nominal;
  * obstacle faces the camera (fronto-parallel) and is static in the world;
  * closing speed is the drone's forward speed toward the obstacle.

Camera frame: +Z forward (range), +X right, +Y up.
Simulator frame (FLU/ENU-like): +x forward, +y left, +z up.
"""
import json
import math
from typing import Optional

from dronesim.config import CameraConfig, ObstacleConfig


def focal_length_px(image_width_px: int, hfov_deg: float) -> float:
    return (image_width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def range_from_box(w_px: float, h_px: float, f_px: float, obs: ObstacleConfig) -> float:
    """Range from the box size; combines the width and height estimates (geometric mean)."""
    z_w = f_px * obs.width_m / w_px
    z_h = f_px * obs.height_m / h_px
    return math.sqrt(z_w * z_h)


def reconstruct_samples(samples: list, info: dict, cam: CameraConfig, obs: ObstacleConfig) -> list:
    """Per-sample metric state. Frames without a detection carry the last range forward at the
    last closing speed (flagged 'extrapolated'); before any detection nothing is reported."""
    f = focal_length_px(info["width"], cam.hfov_deg)
    cx0, cy0 = info["width"] / 2.0, info["height"] / 2.0
    out, last = [], None  # last = (t, Z, v, lateral_right, up)
    for s in samples:
        row = {"frame": s["frame"], "t": s["t"], "source": "none", "range_m": None,
               "closing_speed_mps": None, "lateral_right_m": None, "up_m": None,
               "width_m": None, "height_m": None, "ttc_s": s["ttc_s"]}
        if s["detected"]:
            w, h = s["w_px"], s["h_px"]
            Z = range_from_box(w, h, f, obs)
            x1, y1, x2, y2 = s["bbox"]
            lat = ((x1 + x2) / 2 - cx0) * Z / f
            up = (cy0 - (y1 + y2) / 2) * Z / f
            size = math.sqrt(w * h)
            v = Z * s["growth_px_s"] / size if s["growth_px_s"] is not None else None
            row.update(source="measured", range_m=Z, closing_speed_mps=v, lateral_right_m=lat, up_m=up,
                       width_m=w * Z / f, height_m=h * Z / f)
            if v is not None:
                last = (s["t"], Z, v, lat, up)
            elif last is not None:
                last = (s["t"], Z, last[2], lat, up)
        elif last is not None and s["ttc_source"] == "extrapolated":
            t0, Z0, v0, lat, up = last
            Z = max(0.0, Z0 - v0 * (s["t"] - t0))
            row.update(source="extrapolated", range_m=Z, closing_speed_mps=v0,
                       lateral_right_m=lat, up_m=up, width_m=obs.width_m, height_m=obs.height_m)
        out.append(row)
    return out


def build_scenario(rows: list, obs: ObstacleConfig, n_ref: int = 12) -> Optional[dict]:
    """Static-world scenario at the first reliable estimate.

    Drone starts at the origin facing +x with the measured closing speed (median of the first
    n_ref measured rows that have a speed); the obstacle is a square panel at the measured range.
    """
    ref = [r for r in rows if r["source"] == "measured" and r["closing_speed_mps"] is not None][:n_ref]
    if not ref:
        return None

    def med(k):
        return sorted(r[k] for r in ref)[len(ref) // 2]

    Z, v = med("range_m"), med("closing_speed_mps")
    return {
        "reference_time_s": ref[0]["t"],
        "drone": {"position_m": [0.0, 0.0, 0.0], "velocity_mps": [max(v, 0.0), 0.0, 0.0], "heading": "+x"},
        "obstacle": {
            "center_m": [Z, -med("lateral_right_m"), med("up_m")],  # camera right -> sim -y (left-positive)
            "width_m": obs.width_m, "height_m": obs.height_m, "thickness_m": obs.thickness_m,
            "normal": [-1.0, 0.0, 0.0], "static": True,
        },
        "approach_speed_mps": v,
        "initial_range_m": Z,
        "initial_ttc_s": Z / v if v > 1e-6 else None,
        "assumptions": {"size_is_assumed": obs.size_is_assumed, "fov_is_assumed": None},
    }


def reconstruct_file(track_path: str, out_path: Optional[str] = None) -> dict:
    with open(track_path) as f:
        track = json.load(f)
    cam = CameraConfig(**track["config"]["camera"])
    obs = ObstacleConfig(**track["config"]["obstacle"])
    rows = reconstruct_samples(track["samples"], track["video"], cam, obs)
    scenario = build_scenario(rows, obs)
    if scenario:
        scenario["assumptions"]["fov_is_assumed"] = cam.fov_is_assumed
    result = {"schema": "obstacle_metric/v1",
              "camera": {"hfov_deg": cam.hfov_deg, "focal_px": focal_length_px(track["video"]["width"], cam.hfov_deg)},
              "scenario": scenario, "samples": rows}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
    return result


def main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    src = argv[0]
    dst = argv[1] if len(argv) > 1 else src.replace("obstacle_track", "obstacle_metric")
    sc = reconstruct_file(src, dst)["scenario"]
    print(f"Wrote {dst}")
    if sc:
        print(f"range0={sc['initial_range_m']:.2f} m  approach={sc['approach_speed_mps']:.2f} m/s  "
              f"TTC0={sc['initial_ttc_s']:.1f} s  obstacle={sc['obstacle']['width_m']}x{sc['obstacle']['height_m']} m "
              f"(size assumed={sc['assumptions']['size_is_assumed']})")


if __name__ == "__main__":
    main()
