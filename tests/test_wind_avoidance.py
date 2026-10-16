import json
import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dronesim.api.app import create_app
from dronesim.sim.collision import Box, Environment
from dronesim.sim.params import GRAVITY, DroneParams
from dronesim.sim.planner import (Trajectory, plan_avoidance, required_acceleration, station_keeping_requirement,
                                  trajectory_limits)
from dronesim.sim.recording import Replay, build_recording
from dronesim.sim.runner import RunConfig, fly_trajectory, run_avoidance
from dronesim.sim.wind import WindConfig, wind_vector

P = DroneParams()
K = P.drag_coeff_kg_per_m / P.mass_kg
START = np.array([0.0, 0.0, 1.5])


def scenario(speed=1.5, range_m=2.7, lateral=0.0):
    return {"approach_speed_mps": speed, "reference_time_s": 0.5,
            "drone": {"position_m": [0, 0, 0], "velocity_mps": [speed, 0, 0]},
            "obstacle": {"center_m": [range_m, lateral, 0.0], "width_m": 0.5, "height_m": 0.5, "thickness_m": 0.05,
                         "normal": [-1, 0, 0], "static": True},
            "assumptions": {"size_is_assumed": True, "fov_is_assumed": True}}


def env_for(sc, alt=1.5, margin=0.25):
    return Environment([Box.from_obstacle(sc["obstacle"], alt)], 0.10, margin, 0.0)


@lru_cache(maxsize=None)
def run(speed, az, el, sigma=0.0, tau=2.0, seed=0, replans=2, straight=True):
    cfg = RunConfig(wind=WindConfig(speed, az, el, sigma, tau, seed), max_replans=replans, fly_without_avoidance=straight)
    return run_avoidance(scenario(), P, cfg), cfg


# ------------------------------------------------------------------ planner mathematics
def test_required_acceleration_is_plan_plus_drag_in_air_relative_velocity():
    t = np.linspace(0, 1, 5)
    traj = Trajectory(t, np.zeros((5, 3)), np.tile([1.5, 0.3, -0.2], (5, 1)), np.tile([0.2, 0.0, 0.1], (5, 1)))
    for w in ([0, 0, 0], [-4, 0, 0], [2, -3, 1.5], [1.5, 0.3, -0.2]):
        w = np.array(w, dtype=float)
        v_rel = traj.vel[0] - w
        expected = traj.acc[0] + K * np.linalg.norm(v_rel) * v_rel
        assert required_acceleration(traj, P, w)[0] == pytest.approx(expected, rel=1e-12)
    # flying with the wind (v == w) => no drag => rotors only supply the planned acceleration
    assert required_acceleration(traj, P, traj.vel[0])[0] == pytest.approx(traj.acc[0])


def test_limits_reflect_the_force_balance_including_tilt_and_thrust():
    t = np.linspace(0, 1, 3)
    still = Trajectory(t, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))        # hover
    for w in ([5, 0, 0], [0, -4, 3], [-6, 2, -2]):
        w = np.array(w, dtype=float)
        lim = trajectory_limits(still, P, w)
        f = -K * np.linalg.norm(w) * w + np.array([0, 0, GRAVITY])      # rotors cancel the (downwind) drag
        assert lim["peak_thrust_n"] == pytest.approx(P.mass_kg * np.linalg.norm(f), rel=1e-12)
        assert lim["peak_tilt_deg"] == pytest.approx(math.degrees(math.atan2(np.linalg.norm(f[:2]), f[2])), rel=1e-12)
        assert lim["peak_accel_mps2"] == pytest.approx(K * np.linalg.norm(w) ** 2, rel=1e-12)
        sk = station_keeping_requirement(P, w)
        assert (sk["thrust_n"], sk["tilt_deg"], sk["accel_mps2"]) == pytest.approx(
            (lim["peak_thrust_n"], lim["peak_tilt_deg"], lim["peak_accel_mps2"]))


def test_wind_makes_candidates_harder_and_is_priced_into_the_plan():
    sc = scenario()
    env = env_for(sc)
    calm = plan_avoidance(env, START, 1.5, P)
    windy = plan_avoidance(env, START, 1.5, P, wind=wind_vector(5, 180, 0))
    assert calm.plan and windy.plan
    hard = lambda r: r.rejections["accel"] + r.rejections["thrust"] + r.rejections["tilt"]
    assert hard(windy) > hard(calm)
    assert windy.plan.metrics["peak_tilt_deg"] > calm.plan.metrics["peak_tilt_deg"] + 5
    assert windy.wind_mps == pytest.approx((-5, 0, 0), abs=1e-12)


def test_wind_beyond_authority_is_explained_not_faked():
    sc = scenario()
    r = plan_avoidance(env_for(sc), START, 1.5, P, wind=wind_vector(12, 180, 0))
    assert r.needs_avoidance and r.plan is None
    assert "UNAVOIDABLE" in r.note and "Wind 12.0 m/s" in r.note and "tilt" in r.nominal_violations
    assert r.nominal_limits["peak_tilt_deg"] > P.max_tilt_deg
    clear = plan_avoidance(env_for(scenario(lateral=5.0)), START, 1.5, P, wind=wind_vector(12, 180, 0))
    assert not clear.needs_avoidance and "exceeds the drone's limits" in clear.note


def test_extra_buffer_demands_a_wider_detour():
    env = env_for(scenario())
    a = plan_avoidance(env, START, 1.5, P)
    b = plan_avoidance(env, START, 1.5, P, extra_buffer_m=0.3)
    assert b.plan.params["offset_m"] > a.plan.params["offset_m"]
    assert b.plan.metrics["min_clearance_m"] >= 0.25 + 0.05 + 0.3 - 1e-9


# ------------------------------------------------------------------ closed loop: controller + physics + planner
WINDS = [(4, 180, 0), (4, 0, 0), (4, 90, 0), (4, 0, 90), (4, 0, -90), (5, 135, 30), (5, -60, -40)]


@pytest.mark.parametrize("speed,az,el", WINDS)
def test_avoidance_works_in_any_3d_wind_direction(speed, az, el):
    r, cfg = run(speed, az, el)
    s, n = r.summary, r.nominal_summary
    assert n.collided                                              # straight flight still hits the panel in this wind
    assert not s.collided and s.min_clearance_m >= r.environment.safety_margin_m   # measured on the flown path
    assert s.max_tracking_error_m < 0.15 and s.saturated_fraction == 0.0 and s.limited_fraction == 0.0
    assert r.plan_result.wind_mps == pytest.approx(tuple(wind_vector(speed, az, el)), abs=1e-12)
    h = r.history
    assert h["wind"] == pytest.approx(np.tile(wind_vector(speed, az, el), (len(h["t"]), 1)), abs=1e-12)   # steady wind
    assert h["command"].min() >= 0 and h["command"].max() <= 1


def test_mirror_symmetry_left_and_right_crosswind():
    """Reflecting the world across the x-z plane maps a +y crosswind to a -y one: y -> -y, everything else equal."""
    a, _ = run(4, 90, 0)
    b, _ = run(4, -90, 0)
    pa, pb = a.history["position"], b.history["position"]
    assert pb[:, 1] == pytest.approx(-pa[:, 1], abs=1e-6)
    assert pb[:, [0, 2]] == pytest.approx(pa[:, [0, 2]], abs=1e-6)
    assert a.summary.min_clearance_m == pytest.approx(b.summary.min_clearance_m, abs=1e-6)
    ca, cb = a.history["command"], b.history["command"]
    assert cb == pytest.approx(ca[:, [3, 2, 1, 0]], abs=1e-6)        # M1<->M4 (FR<->FL), M2<->M3 (RR<->RL)


def test_drone_starts_in_equilibrium_with_the_wind_and_observer_matches_analytic_drag():
    """Pre-roll: at t = 0 the controller has learned d = -(c_d/m)|v-w|(v-w) and the drone is on the line."""
    sc = scenario()
    env = env_for(sc)
    w = wind_vector(4, 180, 0)                                       # headwind: v_rel = 1.5 + 4 = 5.5 m/s
    cfg = RunConfig(wind=WindConfig(4, 180, 0))
    nominal = plan_avoidance(env, START, 1.5, P, wind=w).nominal
    h = fly_trajectory(nominal, START, np.array([1.5, 0, 0]), P, cfg, 1.0)
    d_true = -K * 5.5 * np.array([5.5, 0, 0])
    assert h["disturbance_est"][0] == pytest.approx(d_true, rel=0.03, abs=0.02)
    assert np.linalg.norm(h["position"][0] - START) < 0.02
    assert h["velocity"][0] == pytest.approx([1.5, 0, 0], abs=0.02)
    # without the pre-roll the observer starts cold and the same flight sags much more
    cold = fly_trajectory(nominal, START, np.array([1.5, 0, 0]), P, replace(cfg, preroll_s=0.0), 4.0)
    warm = fly_trajectory(nominal, START, np.array([1.5, 0, 0]), P, cfg, 4.0)
    err = lambda hh: np.linalg.norm(hh["position"] - hh["setpoint_position"], axis=1).max()
    assert err(cold) > 3 * err(warm)


def test_strong_wind_is_reported_with_limit_fraction_in_the_flown_summary():
    """Gusts that push the demanded acceleration past the limit must show up in the data, not be hidden."""
    r, _ = run(4, 90, 0, sigma=2.0, tau=1.0, seed=0, replans=0, straight=False)
    assert r.summary.limited_fraction > 0.05 and r.summary.max_tracking_error_m > 0.5


def test_gust_induced_clearance_loss_triggers_a_larger_buffer_replan():
    first, _ = run(4, 90, 0, sigma=2.0, tau=1.0, seed=4, replans=0, straight=False)
    assert first.summary.min_clearance_m < first.environment.safety_margin_m       # the gust ate the margin
    assert len(first.replans) == 1
    r, _ = run(4, 90, 0, sigma=2.0, tau=1.0, seed=4, replans=2, straight=False)
    bufs = [a["extra_buffer_m"] for a in r.replans]
    assert len(bufs) >= 2 and bufs == sorted(bufs) and bufs[0] == 0.0 and bufs[-1] > 0
    assert r.summary.min_clearance_m >= r.environment.safety_margin_m              # verified on the final flown path
    assert r.summary.min_clearance_m > first.summary.min_clearance_m


def test_wind_runs_are_reproducible_and_the_seed_matters():
    a, _ = run(3, 60, 20, sigma=0.8, seed=5, straight=False)
    b = run_avoidance(scenario(), P, RunConfig(wind=WindConfig(3, 60, 20, 0.8, 2.0, 5), fly_without_avoidance=False))
    c = run_avoidance(scenario(), P, RunConfig(wind=WindConfig(3, 60, 20, 0.8, 2.0, 6), fly_without_avoidance=False))
    assert np.array_equal(a.history["position"], b.history["position"])
    assert np.array_equal(a.history["wind"], b.history["wind"])
    assert not np.array_equal(a.history["wind"], c.history["wind"])
    # the two flights in one run (avoiding / straight) see the same gust realisation
    d, _ = run(3, 60, 20, sigma=0.8, seed=5)
    assert np.array_equal(d.history["wind"][:200], d.nominal_history["wind"][:200])


# ------------------------------------------------------------------ recording + replay
def test_recording_carries_wind_description_and_per_sample_wind():
    r, cfg = run(4, 135, 30)
    rec = json.loads(json.dumps(build_recording(r, P, cfg)))
    w = rec["wind"]
    assert w["config"] == {"speed_mps": 4, "azimuth_deg": 135, "elevation_deg": 30, "turbulence_sigma_mps": 0.0,
                           "turbulence_tau_s": 2.0, "seed": 0}
    assert w["mean_vector_mps"] == pytest.approx(wind_vector(4, 135, 30), abs=1e-6)
    c = w["components_approach_frame"]
    assert c["headwind_mps"] == pytest.approx(-4 * math.cos(math.radians(30)) * math.cos(math.radians(135)), abs=1e-6)
    assert c["crosswind_left_mps"] == pytest.approx(4 * math.cos(math.radians(30)) * math.sin(math.radians(135)), abs=1e-6)
    assert c["vertical_up_mps"] == pytest.approx(2.0, abs=1e-6)
    assert w["station_keeping"]["tilt_deg"] == pytest.approx(station_keeping_requirement(P, wind_vector(4, 135, 30))["tilt_deg"])
    assert w["preroll_s"] == 6.0 and "TOWARD" in w["convention"]
    f = rec["flight"]
    for k in ("wind_mps", "limited", "disturbance_est_mps2"):
        assert len(f[k]) == len(f["t"]), k
    assert rec["planning"]["nominal_limits"]["peak_tilt_deg"] > 0 and rec["planning"]["attempts"]
    assert "limited_fraction" in rec["summary"]


def test_replay_exposes_wind_and_air_relative_speed():
    r, cfg = run(4, 135, 30)
    rec = build_recording(r, P, cfg)
    rp = Replay(rec)
    s = rp.at(2.5)
    assert s["wind_mps"] == pytest.approx(wind_vector(4, 135, 30), abs=1e-5)
    assert s["airspeed_mps"] == pytest.approx(np.linalg.norm(s["velocity_mps"] - s["wind_mps"]))
    assert s["limited"] is False


def test_old_recordings_without_wind_columns_still_replay():
    r, cfg = run(4, 135, 30)
    rec = json.loads(json.dumps(build_recording(r, P, cfg)))
    for k in ("wind_mps", "limited", "disturbance_est_mps2"):
        rec["flight"].pop(k)
    s = Replay(rec).at(1.0)
    assert s["wind_mps"] == pytest.approx(0) and s["limited"] is False
    assert s["airspeed_mps"] == pytest.approx(np.linalg.norm(s["velocity_mps"]))


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(tmp_path):
    d = tmp_path / "runs" / "demo"
    d.mkdir(parents=True)
    (d / "obstacle_metric.json").write_text(json.dumps({"scenario": scenario(), "samples": []}))
    return TestClient(create_app(tmp_path / "runs", [], frontend_dist=tmp_path / "none"))


def test_api_wind_parameters_are_applied_and_reported(client):
    r = client.post("/api/runs/demo/simulate", json={"wind_speed_mps": 4, "wind_azimuth_deg": 135, "wind_elevation_deg": 30,
                                                     "turbulence_sigma_mps": 0.0})
    assert r.status_code == 200
    rec = r.json()
    assert rec["wind"]["config"]["speed_mps"] == 4 and rec["wind"]["config"]["azimuth_deg"] == 135
    assert rec["wind"]["config"]["elevation_deg"] == 30
    assert rec["summary"]["collided"] is False and rec["nominal_summary"]["collided"] is True
    direct, cfg = run(4, 135, 30)
    assert rec == json.loads(json.dumps(build_recording(direct, P, cfg, source={"metric_json": "obstacle_metric.json"})))


def test_api_default_is_still_air_and_wind_changes_the_flight(client):
    calm = client.post("/api/runs/demo/simulate", json={"dodge": "left"}).json()
    windy = client.post("/api/runs/demo/simulate", json={"dodge": "left", "wind_speed_mps": 5, "wind_azimuth_deg": 180}).json()
    assert calm["wind"]["config"]["speed_mps"] == 0
    assert np.abs(np.array(calm["flight"]["wind_mps"])).max() == 0
    assert np.array(windy["flight"]["wind_mps"])[:, 0].min() == pytest.approx(-5, abs=1e-6)
    tilt = lambda rec: max(abs(e[1]) for e in rec["flight"]["euler_rpy_deg"])
    assert tilt(windy) > tilt(calm) + 5                              # leaning into a 5 m/s headwind


def test_api_validates_wind_inputs(client):
    for bad in ({"wind_speed_mps": -1}, {"wind_speed_mps": 99}, {"wind_elevation_deg": 120}, {"wind_elevation_deg": -91},
                {"turbulence_sigma_mps": -1}, {"turbulence_sigma_mps": 50}, {"turbulence_tau_s": 0.0},
                {"wind_seed": -3}, {"wind_azimuth_deg": "north"}):
        assert client.post("/api/runs/demo/simulate", json=bad).status_code == 422, bad
