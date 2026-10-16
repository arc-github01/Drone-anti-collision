"""Obstacle-state estimation from a tracked bounding-box time series (pixel units only).

Pure NumPy, no YOLO/OpenCV dependency. Metric reconstruction lives elsewhere.

Time-to-contact model: for a camera closing on an object at roughly constant speed,
apparent size h is proportional to 1/Z, so 1/h is linear in time:
    1/h(t) = (Z0 - v*t) / k      =>     TTC = Z/v = -(1/h) / d(1/h)/dt
A least-squares line over a short window gives d(1/h)/dt, which is much less noisy than
differencing two box heights.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

from dronesim.config import VisionConfig


@dataclass
class Observation:
    frame: int
    t: float                      # seconds since video start (frame / fps)
    bbox: Optional[tuple]         # (x1, y1, x2, y2) pixels, None if not detected
    conf: float = 0.0


@dataclass
class ObstacleSample:
    frame: int
    t: float
    detected: bool
    conf: float
    bbox: Optional[list]
    w_px: Optional[float]
    h_px: Optional[float]
    growth_px_s: Optional[float]  # d(size)/dt of sqrt(w*h), from the fit
    ttc_s: Optional[float]
    ttc_source: str               # "fit" | "extrapolated" | "none"

    def to_dict(self):
        return asdict(self)


def _fit_ttc(ts: np.ndarray, sizes: np.ndarray, max_ttc_s: float):
    """Return (growth_px_s, ttc_s) from a line fit of 1/size vs t; ttc None if not approaching."""
    inv = 1.0 / sizes
    slope, intercept = np.polyfit(ts, inv, 1)
    t_now = ts[-1]
    inv_now = slope * t_now + intercept
    # d(size)/dt = -slope / inv^2
    growth = -slope / inv_now**2 if inv_now > 0 else None
    if slope >= 0 or inv_now <= 0:
        return growth, None
    ttc = float(-inv_now / slope)
    return growth, (ttc if ttc <= max_ttc_s else None)


def estimate_obstacle_states(
    observations: Sequence[Observation], cfg: VisionConfig = VisionConfig()
) -> list[ObstacleSample]:
    """One ObstacleSample per observation, in order. Gaps are extrapolated up to cfg.max_gap_s."""
    out: list[ObstacleSample] = []
    win_t: list[float] = []
    win_s: list[float] = []
    last_ttc = None            # (ttc, t) of the last fitted estimate
    last_growth = None
    last_seen_t = None

    for ob in observations:
        if ob.bbox is not None:
            x1, y1, x2, y2 = ob.bbox
            w, h = float(x2 - x1), float(y2 - y1)
            win_t.append(ob.t)
            win_s.append(float(np.sqrt(max(w * h, 1e-9))))
            while win_t and ob.t - win_t[0] > cfg.fit_window_s:
                win_t.pop(0)
                win_s.pop(0)
            last_seen_t = ob.t
            ttc, growth, src = None, None, "none"
            if len(win_t) >= cfg.min_fit_samples:
                growth, ttc = _fit_ttc(np.array(win_t), np.array(win_s), cfg.max_ttc_s)
                if ttc is not None:
                    src = "fit"
                    last_ttc = (ttc, ob.t)
                    last_growth = growth
            out.append(ObstacleSample(ob.frame, ob.t, True, ob.conf, [float(v) for v in ob.bbox],
                                      w, h, growth, ttc, src))
        else:
            ttc, growth, src = None, None, "none"
            if last_ttc is not None and ob.t - last_seen_t <= cfg.max_gap_s:
                ttc = max(0.0, last_ttc[0] - (ob.t - last_ttc[1]))
                growth, src = last_growth, "extrapolated"
            out.append(ObstacleSample(ob.frame, ob.t, False, 0.0, None, None, None, growth, ttc, src))
    return out
