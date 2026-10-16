"""Select the obstacle track, estimate its state and write obstacle_track.json.

Usage:  python -m dronesim.vision.export VIDEO [--out DIR] [--track-id N]
        [--width-m 0.5 --height-m 0.5] [--hfov-deg 82.6]
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Optional, Sequence

from dronesim.config import PipelineConfig
from dronesim.vision.detector import Detection, track_video, video_info
from dronesim.vision.obstacle_estimator import Observation, estimate_obstacle_states


def select_target_track(frames: Sequence[Sequence[Detection]]) -> Optional[int]:
    """Track with the largest accumulated box area (a looming obstacle dominates the view)."""
    area = defaultdict(float)
    for dets in frames:
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            area[d.track_id] += (x2 - x1) * (y2 - y1)
    return max(area, key=area.get) if area else None


def other_tracks(frames: Sequence[Sequence[Detection]], target_id: int, min_frames: int = 10) -> list:
    """Summary of every OTHER tracked object (not used as the obstacle), so the UI can say what else was seen."""
    stats = {}
    for i, dets in enumerate(frames):
        for d in dets:
            if d.track_id == target_id:
                continue
            x1, y1, x2, y2 = d.bbox
            s = stats.setdefault(d.track_id, {"names": [], "first": i, "last": i, "n": 0, "max_area": 0.0})
            s["names"].append(d.cls_name)
            s["last"] = i
            s["n"] += 1
            s["max_area"] = max(s["max_area"], (x2 - x1) * (y2 - y1))
    out = [{"track_id": t, "yolo_class": max(set(s["names"]), key=s["names"].count), "frames": s["n"],
            "first_frame": s["first"], "last_frame": s["last"], "max_area_px": round(s["max_area"])}
           for t, s in stats.items() if s["n"] >= min_frames]
    return sorted(out, key=lambda r: -r["frames"])


def build_track_json(frames, info: dict, cfg: PipelineConfig, track_id: Optional[int] = None) -> dict:
    tid = track_id if track_id is not None else select_target_track(frames)
    if tid is None:
        raise ValueError("No detections found; lower --conf or check the video/classes.")
    obs, names = [], []
    for i, dets in enumerate(frames):
        match = next((d for d in dets if d.track_id == tid), None)
        if match:
            names.append(match.cls_name)
        obs.append(Observation(i, i / info["fps"], match.bbox if match else None, match.conf if match else 0.0))
    samples = estimate_obstacle_states(obs, cfg.vision)
    return {
        "schema": "obstacle_track/v1",
        "video": info,
        "config": asdict(cfg),
        "target": {"track_id": tid, "yolo_class": max(set(names), key=names.count) if names else None,
                   "detected_frames": sum(s.detected for s in samples)},
        "other_tracks": other_tracks(frames, tid),
        "notes": ["w_px/h_px are PIXELS, not metres.",
                  "Only the target track is used as the obstacle; other_tracks lists what else YOLO followed.",
                  "Metric size/range require obstacle.width_m/height_m and camera hfov (assumed, see config)."],
        "samples": [s.to_dict() for s in samples],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output dir (default data/runs/<video name>)")
    ap.add_argument("--track-id", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--width-m", type=float, default=None)
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--hfov-deg", type=float, default=None)
    a = ap.parse_args(argv)

    cfg = PipelineConfig()
    if a.conf is not None: cfg.vision.conf = a.conf
    if a.width_m is not None: cfg.obstacle.width_m = a.width_m; cfg.obstacle.size_is_assumed = False
    if a.height_m is not None: cfg.obstacle.height_m = a.height_m; cfg.obstacle.size_is_assumed = False
    if a.hfov_deg is not None: cfg.camera.hfov_deg = a.hfov_deg; cfg.camera.fov_is_assumed = False

    info = video_info(a.video)
    frames = list(track_video(a.video, cfg.vision, info["fps"]))
    data = build_track_json(frames, info, cfg, a.track_id)
    out = a.out or os.path.join("data", "runs", os.path.splitext(os.path.basename(a.video))[0])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "obstacle_track.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"Wrote {path}: track {data['target']['track_id']} ({data['target']['yolo_class']}), "
          f"{data['target']['detected_frames']}/{len(frames)} frames detected")


if __name__ == "__main__":
    main()
