import numpy as np
import pytest

from dronesim.config import VisionConfig
from dronesim.vision.obstacle_estimator import Observation, estimate_obstacle_states

F_PX, W_M = 800.0, 0.5


def box_at(Z, cx=640, cy=360):
    s = F_PX * W_M / Z
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def approach(Z0=10.0, v=2.0, fps=24, dur=3.0, drop=None):
    obs = []
    for i in range(int(dur * fps)):
        t = i / fps
        Z = Z0 - v * t
        bb = None if (drop and drop[0] <= i < drop[1]) else box_at(Z)
        obs.append(Observation(i, t, bb, 0.8 if bb else 0.0))
    return obs


def test_ttc_matches_ground_truth_constant_closing_speed():
    s = estimate_obstacle_states(approach())
    for smp in s[10:]:
        true_ttc = (10.0 - 2.0 * smp.t) / 2.0
        assert smp.ttc_source == "fit"
        assert smp.ttc_s == pytest.approx(true_ttc, rel=0.05)


def test_growth_is_positive_when_approaching():
    s = estimate_obstacle_states(approach())
    assert all(x.growth_px_s > 0 for x in s[10:])


def test_receding_or_static_gives_no_ttc():
    for v in (0.0, -1.0):
        s = estimate_obstacle_states(approach(v=v, dur=2.0))
        assert all(x.ttc_s is None for x in s)


def test_too_few_samples_gives_no_ttc():
    s = estimate_obstacle_states(approach(dur=0.15))
    assert all(x.ttc_s is None for x in s)


def test_dropout_extrapolates_then_expires():
    cfg = VisionConfig(max_gap_s=1.0)
    s = estimate_obstacle_states(approach(Z0=20.0, dur=3.0, drop=(36, 72)), cfg)
    gap = s[40]
    assert not gap.detected and gap.ttc_source == "extrapolated"
    true_ttc = (20.0 - 2.0 * gap.t) / 2.0
    assert gap.ttc_s == pytest.approx(true_ttc, rel=0.1)
    assert s[70].ttc_source == "none"   # more than 1 s after last detection


def test_noise_robustness():
    rng = np.random.default_rng(0)
    obs = approach()
    noisy = [Observation(o.frame, o.t, tuple(np.array(o.bbox) + rng.normal(0, 1.0, 4)), o.conf) for o in obs]
    s = estimate_obstacle_states(noisy)
    last = s[-1]
    assert last.ttc_s == pytest.approx((10.0 - 2.0 * last.t) / 2.0, abs=0.25)


def test_select_target_and_build_json_without_yolo():
    from dronesim.config import PipelineConfig
    from dronesim.vision.detector import Detection
    from dronesim.vision.export import build_track_json, select_target_track

    info = dict(path="x.mp4", fps=24.0, width=1280, height=720, n_frames=72)
    frames = []
    for i in range(72):
        bb = box_at(10.0 - 2.0 * i / 24)
        dets = [Detection(i, i / 24, 7, "box", 0.9, bb), Detection(i, i / 24, 9, "car", 0.9, (0, 0, 20, 20))]
        frames.append(dets)
    assert select_target_track(frames) == 7
    data = build_track_json(frames, info, PipelineConfig())
    assert data["target"]["track_id"] == 7 and len(data["samples"]) == 72
    others = data["other_tracks"]                                   # the second object is reported, not used
    assert [o["track_id"] for o in others] == [9] and others[0]["yolo_class"] == "car" and others[0]["frames"] == 72
    assert data["samples"][-1]["ttc_s"] == pytest.approx((10 - 2 * 71 / 24) / 2, rel=0.05)
    import json; json.dumps(data)  # serialisable
