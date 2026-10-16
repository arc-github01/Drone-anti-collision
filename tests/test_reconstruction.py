import math

import pytest

from dronesim.config import CameraConfig, ObstacleConfig
from dronesim.vision.obstacle_estimator import Observation, estimate_obstacle_states
from dronesim.vision.reconstruction import (build_scenario, focal_length_px, range_from_box,
                                            reconstruct_samples)

INFO = dict(width=1280, height=720, fps=24.0)
CAM = CameraConfig(hfov_deg=82.6)
OBS = ObstacleConfig(width_m=0.6, height_m=0.4)
F = focal_length_px(1280, 82.6)


def box(Z, X=0.0, Y=0.0):
    w, h = F * OBS.width_m / Z, F * OBS.height_m / Z
    cx, cy = 640 + F * X / Z, 360 - F * Y / Z
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(Z0=6.0, v=1.5, X=0.3, Y=-0.2, dur=3.0, drop=None):
    obs = []
    for i in range(int(dur * 24)):
        t = i / 24
        bb = None if drop and drop[0] <= i < drop[1] else box(Z0 - v * t, X, Y)
        obs.append(Observation(i, t, bb, 0.9 if bb else 0.0))
    samples = [s.to_dict() for s in estimate_obstacle_states(obs)]
    return reconstruct_samples(samples, INFO, CAM, OBS)


def test_focal_length_formula():
    assert F == pytest.approx(640 / math.tan(math.radians(41.3)))


def test_range_recovers_ground_truth_for_both_dimensions():
    for Z in (2.0, 5.0, 12.0):
        b = box(Z)
        assert range_from_box(b[2] - b[0], b[3] - b[1], F, OBS) == pytest.approx(Z, rel=1e-9)


def test_range_closing_speed_and_offsets():
    rows = run()
    for r in rows[12:]:
        assert r["range_m"] == pytest.approx(6.0 - 1.5 * r["t"], rel=1e-6)
        assert r["closing_speed_mps"] == pytest.approx(1.5, rel=0.06)
        assert r["lateral_right_m"] == pytest.approx(0.3, abs=1e-6)
        assert r["up_m"] == pytest.approx(-0.2, abs=1e-6)
        assert r["width_m"] == pytest.approx(0.6, rel=1e-6)


def test_dropout_is_extrapolated_with_correct_range():
    rows = run(Z0=8.0, drop=(36, 54))
    g = rows[45]
    assert g["source"] == "extrapolated"
    assert g["range_m"] == pytest.approx(8.0 - 1.5 * g["t"], rel=0.05)


def test_scenario_geometry_and_frame_convention():
    sc = build_scenario(run(), OBS)
    assert sc["drone"]["velocity_mps"][0] == pytest.approx(1.5, rel=0.06)
    cx, cy, cz = sc["obstacle"]["center_m"]
    assert cx == pytest.approx(sc["initial_range_m"])
    assert cy < 0 and cz < 0   # obstacle is right of the axis (-y, left-positive) and below it
    assert sc["initial_ttc_s"] == pytest.approx(sc["initial_range_m"] / sc["approach_speed_mps"])


def test_no_detections_gives_no_scenario():
    rows = reconstruct_samples([dict(frame=0, t=0.0, detected=False, ttc_s=None, ttc_source="none")], INFO, CAM, OBS)
    assert build_scenario(rows, OBS) is None
