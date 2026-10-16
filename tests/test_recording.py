import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dronesim.sim.params import DroneParams
from dronesim.sim.recording import Replay, build_recording, load_recording, save_recording
from dronesim.sim.runner import RunConfig, run_avoidance

P = DroneParams()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scenario(speed=1.5, range_m=2.7):
    return {"approach_speed_mps": speed, "reference_time_s": 0.5,
            "drone": {"position_m": [0, 0, 0], "velocity_mps": [speed, 0, 0]},
            "obstacle": {"center_m": [range_m, 0.0, 0.0], "width_m": 0.5, "height_m": 0.5,
                         "thickness_m": 0.05, "normal": [-1, 0, 0], "static": True},
            "assumptions": {"size_is_assumed": True, "fov_is_assumed": True}}


@pytest.fixture(scope="module")
def built():
    cfg = RunConfig()
    run = run_avoidance(scenario(), P, cfg)
    return run, cfg, build_recording(run, P, cfg, source={"metric_json": "x.json"})


@pytest.fixture(scope="module")
def rec(built, tmp_path_factory):
    path = tmp_path_factory.mktemp("rec") / "sim_recording.json"
    save_recording(built[2], str(path))
    return load_recording(str(path))


# ------------------------------------------------------------------ file round trip
def test_round_trip_preserves_everything_to_rounding(built, rec):
    orig = built[2]
    assert rec["schema"] == "sim_recording/v1"
    assert json.loads(json.dumps(orig)) == rec                     # identical after a JSON round trip
    f = rec["flight"]
    assert np.allclose(f["position_m"], built[0].history["position"], atol=1e-6)
    assert np.allclose(f["motor_command"], built[0].history["command"], atol=1e-6)


def test_columns_are_consistent_and_time_is_fixed_step(rec):
    f, dt = rec["flight"], rec["physics"]["sample_dt_s"]
    n = len(f["t"])
    for k in ("position_m", "velocity_mps", "quaternion_wxyz", "body_rates_radps", "motor_command",
              "motor_omega_radps", "motor_thrust_n", "motor_power_w", "clearance_m", "saturated"):
        assert len(f[k]) == n, k
    assert np.array(f["motor_command"]).shape == (n, 4)
    assert np.allclose(np.diff(f["t"]), dt, atol=1e-6)             # fixed timestep, no render-rate dependence
    assert dt == pytest.approx(0.004)


def test_rejects_unknown_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "other/v9"}))
    with pytest.raises(ValueError):
        load_recording(str(p))


def test_recording_is_deterministic(built):
    run, cfg, rec1 = built
    assert json.dumps(build_recording(run, P, cfg, source={"metric_json": "x.json"})) == json.dumps(rec1)
    run2 = run_avoidance(scenario(), P, cfg)
    assert json.dumps(build_recording(run2, P, cfg, source={"metric_json": "x.json"})) == json.dumps(rec1)


# ------------------------------------------------------------------ recorded data is physically consistent
def test_recorded_quantities_are_mutually_consistent(rec):
    f, d = rec["flight"], rec["drone"]
    om = np.array(f["motor_omega_radps"])
    assert np.allclose(f["motor_thrust_n"], d["k_thrust"] * om**2, rtol=1e-4, atol=1e-6)
    p = d["params"]
    assert np.allclose(f["motor_power_w"], d["k_torque"] * om**3 / p["motor_efficiency"] + p["motor_idle_power_w"],
                       rtol=1e-4, atol=1e-5)
    # velocity is the derivative of the recorded position (not an independently scripted curve)
    t, pos, vel = np.array(f["t"]), np.array(f["position_m"]), np.array(f["velocity_mps"])
    assert np.abs(np.gradient(pos, t, axis=0)[2:-2] - vel[2:-2]).max() < 0.02
    # omega never exceeds the rotor limit; commands stay in [0,1]
    assert om.max() <= p["omega_max_radps"] + 1e-3 and om.min() >= -1e-6
    assert np.array(f["motor_command"]).min() >= 0 and np.array(f["motor_command"]).max() <= 1
    # quaternions are unit
    assert np.allclose(np.linalg.norm(f["quaternion_wxyz"], axis=1), 1.0, atol=1e-5)


def test_clearance_recomputed_from_recorded_geometry_matches(rec):
    assert np.allclose(Replay(rec).recompute_clearance(), rec["flight"]["clearance_m"], atol=2e-6)
    assert min(rec["flight"]["clearance_m"]) == pytest.approx(rec["summary"]["min_clearance_m"], abs=2e-6)
    assert rec["summary"]["collided"] is False
    assert rec["nominal_summary"]["collided"] is True


def test_planning_and_trajectories_are_stored(rec):
    pl = rec["planning"]
    assert pl["needs_avoidance"] and pl["plan"]["kind"] in ("detour", "brake")
    assert pl["n_candidates"] > 0 and "accel" in pl["rejections"]
    assert len(pl["nominal_trajectory"]["position_m"]) > 10 and len(pl["plan"]["trajectory"]["position_m"]) > 10
    assert rec["environment"]["obstacles"][0]["half_extents_m"] == pytest.approx([0.025, 0.25, 0.25])
    assert any("ASSUMED" in a for a in rec["assumptions"])


# ------------------------------------------------------------------ replay / scrubbing
def test_replay_returns_stored_values_at_sample_times(rec):
    r = Replay(rec)
    f = rec["flight"]
    for i in (0, 1, 137, len(f["t"]) - 1):
        s = r.at(f["t"][i])
        assert s["index"] == i
        assert s["position_m"] == pytest.approx(f["position_m"][i])
        assert s["motor_command"] == pytest.approx(f["motor_command"][i])
        assert s["clearance_m"] == pytest.approx(f["clearance_m"][i])


def test_replay_interpolates_between_samples_and_clamps(rec):
    r, f = Replay(rec), rec["flight"]
    t = (f["t"][10] + f["t"][11]) / 2
    s = r.at(t)
    assert s["position_m"] == pytest.approx((np.array(f["position_m"][10]) + np.array(f["position_m"][11])) / 2)
    assert s["motor_command"] == pytest.approx(f["motor_command"][10])            # zero-order hold
    assert np.linalg.norm(s["quaternion_wxyz"]) == pytest.approx(1.0)
    assert r.at(-5.0)["index"] == 0 and r.at(1e6)["index"] == len(f["t"]) - 1
    assert r.at(1e6)["t"] == pytest.approx(r.duration)


def test_scrubbing_is_order_independent(rec):
    """at(t) is a pure function of t: scrubbing forward, backward or randomly gives identical states."""
    r = Replay(rec)
    ts = np.linspace(0, r.duration, 37)
    fwd = [r.at(t) for t in ts]
    rng = np.random.default_rng(1)
    order = rng.permutation(len(ts))
    shuffled = {int(i): r.at(ts[i]) for i in order}
    back = [r.at(t) for t in ts[::-1]][::-1]
    for i in range(len(ts)):
        for k in ("position_m", "motor_command", "euler_rpy_deg", "velocity_mps"):
            assert np.array_equal(fwd[i][k], shuffled[i][k]) and np.array_equal(fwd[i][k], back[i][k])


def test_motor_series_for_graphs_are_the_controller_outputs(rec, built):
    r, hist = Replay(rec), built[0].history
    for m in range(4):
        t, u = r.motor_series(m)
        assert len(t) == len(u) == len(hist["t"])
        assert u == pytest.approx(hist["command"][:, m], abs=1e-6)
    # the four motors are NOT identical during the manoeuvre (real differential command)
    allu = np.array([r.motor_series(m)[1] for m in range(4)])
    assert (allu.max(axis=0) - allu.min(axis=0)).max() > 1e-3
    # graph value at the playhead equals the replay state's command
    assert Replay(rec).at(1.0)["motor_command"] == pytest.approx(allu[:, r.index_at(1.0)])


def test_path_trail_and_video_sync(rec):
    r = Replay(rec)
    assert len(r.path()) == len(rec["flight"]["t"])
    assert len(r.path(1.0)) == r.index_at(1.0) + 1
    assert r.video_time(0.0) == pytest.approx(0.5)               # scenario reference_time_s
    assert r.video_time(2.0) == pytest.approx(2.5)               # time_scale 1: no speed override


def test_video_sync_is_scaled_when_speed_is_overridden():
    cfg = RunConfig(approach_speed_mps=3.0, fly_without_avoidance=False)
    run = run_avoidance(scenario(speed=1.5), P, cfg)
    r = Replay(build_recording(run, P, cfg))
    assert r.video_time(2.0) == pytest.approx(0.5 + 2.0 * 0.5)   # 1.5/3.0 video seconds per sim second


def test_nominal_flight_can_be_replayed_too(rec):
    r = Replay(rec, "nominal_flight")
    assert r.recompute_clearance().min() < 0
    # this flight penetrates the panel, so it exercises the inside-the-box branch of the geometry too
    assert np.allclose(r.recompute_clearance(), rec["nominal_flight"]["clearance_m"], atol=2e-6)
    assert min(rec["nominal_flight"]["clearance_m"]) == pytest.approx(rec["nominal_summary"]["min_clearance_m"], abs=2e-6)
    with pytest.raises(ValueError):
        Replay({**rec, "nominal_flight": None}, "nominal_flight")


def test_unavoidable_run_records_without_flight():
    cfg = RunConfig(fly_without_avoidance=False)
    run = run_avoidance(scenario(speed=8.0), P, cfg)
    rec = build_recording(run, P, cfg)
    assert rec["flight"] is None and rec["planning"]["plan"] is None and "UNAVOIDABLE" in rec["planning"]["note"]
    json.dumps(rec)
    with pytest.raises(ValueError):
        Replay(rec)


def test_replay_needs_no_yolo_opencv_or_physics():
    code = ("import sys; sys.path.insert(0, %r); import dronesim.sim.recording as r; "
            "bad = [m for m in ('ultralytics','cv2','torch','dronesim.sim.quadrotor','dronesim.sim.planner',"
            "'dronesim.sim.controller') if m in sys.modules]; assert not bad, bad; print('ok')" % ROOT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
