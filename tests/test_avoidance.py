from dataclasses import replace

import numpy as np
import pytest

from dronesim.sim.collision import Box, Environment, predict_collision
from dronesim.sim.params import DroneParams
from dronesim.sim.planner import PlannerConfig, plan_avoidance, trajectory_limits
from dronesim.sim.runner import RunConfig, run_avoidance

P = DroneParams()


def scenario(range_m=2.7, speed=1.5, w=0.5, h=0.5, lateral=0.0, up=0.0):
    return {"approach_speed_mps": speed,
            "drone": {"position_m": [0, 0, 0], "velocity_mps": [speed, 0, 0]},
            "obstacle": {"center_m": [range_m, lateral, up], "width_m": w, "height_m": h,
                         "thickness_m": 0.05, "normal": [-1, 0, 0], "static": True}}


def env_for(sc, alt=1.5, floor=0.0, margin=0.25, radius=0.10):
    return Environment([Box.from_obstacle(sc["obstacle"], alt)], radius, margin, floor)


START = np.array([0.0, 0.0, 1.5])


# ------------------------------------------------------------------ collision geometry
def test_box_signed_distance_faces_edges_corners_inside():
    b = Box(np.zeros(3), np.array([1.0, 1.0, 1.0]))
    sd = lambda *p: float(b.signed_distance(np.array([p]))[0])
    assert sd(3, 0, 0) == pytest.approx(2.0)                       # face
    assert sd(2, 2, 0) == pytest.approx(np.sqrt(2))                # edge
    assert sd(2, 2, 2) == pytest.approx(np.sqrt(3))                # corner
    assert sd(0, 0, 0) == pytest.approx(-1.0)                      # centre: penetration depth
    assert sd(0.5, 0, 0) == pytest.approx(-0.5)
    assert sd(1, 0, 0) == pytest.approx(0.0)                       # on the surface


def test_clearance_includes_drone_radius_and_floor():
    env = Environment([Box(np.array([5.0, 0, 1.0]), np.array([0.05, 1, 1]))], 0.1, 0.25, 0.0)
    assert env.obstacle_clearance(np.array([[3.0, 0, 1.0]]))[0] == pytest.approx(5 - 0.05 - 3 - 0.1)
    assert env.floor_clearance(np.array([[0, 0, 0.5]]))[0] == pytest.approx(0.4)
    assert Environment([env.boxes[0]], floor_z_m=None).floor_clearance(np.array([[0, 0, -9.0]]))[0] == np.inf


def test_predict_collision_reports_first_contact_and_margin_violation():
    env = env_for(scenario())
    t = np.arange(0, 4, 0.01)
    pts = START + np.outer(t, [1.5, 0, 0])
    chk = predict_collision(env, t, pts)
    assert chk["collides"] and chk["violates_margin"]
    assert chk["t_first_violation_s"] < chk["t_first_contact_s"] < 2.7 / 1.5 + 0.1
    side = pts + [0, 3.0, 0]
    chk2 = predict_collision(env, t, side)
    assert not chk2["collides"] and not chk2["violates_margin"] and chk2["min_clearance_m"] > 1


# ------------------------------------------------------------------ planner
def test_clear_path_needs_no_avoidance():
    sc = scenario(lateral=5.0)
    r = plan_avoidance(env_for(sc), START, 1.5, P)
    assert not r.needs_avoidance and r.plan is None


def test_detour_is_feasible_smooth_and_returns_to_line():
    sc = scenario()
    r = plan_avoidance(env_for(sc), START, 1.5, P)
    assert r.needs_avoidance and r.plan and r.plan.kind == "detour"
    tr, m = r.plan.trajectory, r.plan.metrics
    # constraints actually satisfied (with the 80 % fraction)
    assert m["peak_accel_mps2"] <= 0.8 * P.max_accel_mps2
    assert m["peak_thrust_n"] <= 0.8 * 4 * P.max_thrust_per_motor_n
    assert m["peak_tilt_deg"] <= 0.8 * P.max_tilt_deg
    assert m["min_clearance_m"] >= 0.25 + PlannerConfig().tracking_buffer_m - 1e-9
    # starts on the nominal line with nominal velocity and no acceleration jump
    assert tr.pos[0] == pytest.approx(START) and tr.vel[0] == pytest.approx([1.5, 0, 0])
    assert tr.acc[0] == pytest.approx(0, abs=1e-9)
    # ends back on the original line
    assert tr.pos[-1][1:] == pytest.approx(START[1:], abs=1e-6)
    # analytic derivatives are consistent with finite differences of the position
    dt = tr.t[1] - tr.t[0]
    assert np.gradient(tr.pos, dt, axis=0)[5:-5] == pytest.approx(tr.vel[5:-5], abs=2e-3)
    # (acceleration is continuous but jerk jumps where a phase starts/ends, so a numerical derivative
    #  errs by ~jerk*dt at those few samples: require agreement everywhere else, loose bound at them)
    err = np.abs(np.gradient(tr.vel, dt, axis=0) - tr.acc)[5:-5].max(axis=1)
    assert np.mean(err < 5e-3) > 0.95 and err.max() < 0.05
    assert trajectory_limits(tr, P)["peak_accel_mps2"] == pytest.approx(m["peak_accel_mps2"])


def test_infeasible_candidates_are_rejected_with_reasons():
    sc = scenario()
    tight = replace(P, max_accel_mps2=0.05)           # no sidestep can be this gentle at this range
    r = plan_avoidance(env_for(sc), START, 1.5, tight)
    assert r.plan is None and "UNAVOIDABLE" in r.note
    assert r.n_feasible == 0 and r.rejections["accel"] > 0 and r.rejections["clearance"] > 0


def test_too_fast_is_reported_unavoidable_not_faked():
    sc = scenario()
    r = plan_avoidance(env_for(sc), START, 8.0, P)
    assert r.needs_avoidance and r.plan is None and r.n_feasible == 0


def test_floor_is_respected():
    sc = scenario()
    env = env_for(sc, alt=0.9)                        # near the floor: downward dodges must be rejected
    r = plan_avoidance(env, np.array([0, 0, 0.9]), 1.5, P)
    assert r.plan is not None
    assert env.floor_clearance(r.plan.trajectory.pos).min() >= 0.25
    assert r.rejections["floor"] > 0


def test_brake_fallback_when_no_detour_fits():
    sc = scenario(w=30.0, h=30.0, speed=0.5)          # wall too big to sidestep within the offset grid
    r = plan_avoidance(env_for(sc, alt=15.0), np.array([0, 0, 15.0]), 0.5, P)
    assert r.plan is not None and r.plan.kind == "brake"
    tr = r.plan.trajectory
    assert tr.vel[-1] == pytest.approx(0, abs=1e-9)
    assert r.plan.metrics["min_clearance_m"] >= 0.25
    assert tr.pos[-1][0] < 2.7 - 0.025 - 0.1 - 0.25 + 1e-6


# ------------------------------------------------------------------ closed-loop (controller + physics)
def test_flown_avoidance_clears_obstacle_that_straight_flight_hits():
    r = run_avoidance(scenario(speed=1.5))
    assert r.plan_result.plan is not None
    assert r.nominal_summary.collided and r.nominal_summary.min_clearance_m < 0
    s = r.summary
    assert not s.collided and s.min_clearance_m >= r.environment.safety_margin_m     # measured, not planned
    assert s.max_tilt_deg <= P.max_tilt_deg and s.saturated_fraction == 0.0
    assert s.max_tracking_error_m < 0.1
    h = r.history
    assert h["command"].min() >= 0 and h["command"].max() <= 1
    assert h["command"].std(axis=1).max() > 1e-3          # motors actually differ: real differential thrust


def test_flown_avoidance_at_video_scale_speed():
    r = run_avoidance(scenario(range_m=2.72, speed=0.33))
    assert r.nominal_summary.collided
    assert not r.summary.collided and r.summary.min_clearance_m >= r.environment.safety_margin_m


def test_unavoidable_case_flies_nothing_and_says_so():
    r = run_avoidance(scenario(speed=8.0), cfg=RunConfig(fly_without_avoidance=False))
    assert r.plan_result.plan is None and r.history is None and r.summary is None


def test_run_is_deterministic():
    a = run_avoidance(scenario(speed=1.5), cfg=RunConfig(fly_without_avoidance=False))
    b = run_avoidance(scenario(speed=1.5), cfg=RunConfig(fly_without_avoidance=False))
    assert np.array_equal(a.history["position"], b.history["position"])
    assert np.array_equal(a.history["command"], b.history["command"])
