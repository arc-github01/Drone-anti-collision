import math

import numpy as np
import pytest

from dronesim.sim import motors, quaternion as quat
from dronesim.sim.controller import FlightController, Setpoint, desaturate
from dronesim.sim.loop import simulate
from dronesim.sim.params import DroneParams
from dronesim.sim.quadrotor import Quadrotor

P = DroneParams()
TMAX = P.max_thrust_per_motor_n


def tilt_deg(qs):
    """Angle between body z and world z for each quaternion."""
    return np.array([math.degrees(math.acos(np.clip(quat.to_rotation_matrix(q)[2, 2], -1, 1))) for q in qs])


def fly(setpoint_fn, seconds, **reset):
    quad = Quadrotor(P)
    quad.reset(**reset)
    return simulate(quad, FlightController(P), setpoint_fn, seconds)


def hold(pos, yaw=0.0):
    return lambda t: Setpoint(np.array(pos, dtype=float), yaw=yaw)


# ------------------------------------------------------------------ mixer / desaturation
def test_feasible_wrench_is_unchanged():
    tau = np.array([2e-3, -1e-3, 5e-4])
    T, tot, tq, sat = desaturate(P.weight_n, tau, P)
    assert not sat and tot == pytest.approx(P.weight_n) and tq == pytest.approx(tau)
    assert motors.mixer_matrix(P) @ T == pytest.approx([P.weight_n, *tau])


def test_desaturation_respects_limits_and_sacrifices_yaw_first():
    tau = np.array([3e-3, 0.0, 2.5e-2])          # yaw request too large to fit
    T, tot, tq, sat = desaturate(P.weight_n, tau, P)
    assert sat and T.min() >= -1e-12 and T.max() <= TMAX + 1e-12
    assert tq[0] == pytest.approx(3e-3) and abs(tq[2]) < abs(tau[2])


def test_attitude_torque_has_priority_over_thrust():
    T, tot, tq, sat = desaturate(4 * TMAX, np.array([4e-3, 0, 0]), P)   # full thrust + roll: must cut thrust
    assert sat and tot < 4 * TMAX and T.max() <= TMAX + 1e-12
    assert tq[0] == pytest.approx(4e-3)


# ------------------------------------------------------------------ closed loop
def test_hover_hold_is_stable():
    h = fly(hold([0, 0, 5]), 8.0, position=(0, 0, 5))
    assert np.abs(h["position"] - [0, 0, 5]).max() < 1e-4
    assert h["command"].min() > 0 and h["command"].max() < 1
    assert h["command"][-1] == pytest.approx(np.full(4, P.hover_command), abs=1e-3)


def test_position_step_response_settles_with_small_overshoot_and_bounded_tilt():
    h = fly(hold([2.0, 0, 5]), 12.0, position=(0, 0, 5))
    x = h["position"][:, 0]
    assert x[-1] == pytest.approx(2.0, abs=0.02)
    assert x.max() < 2.0 * 1.15                              # <15 % overshoot
    assert np.abs(h["position"][-500:, 1:] - [0, 5]).max() < 0.02
    assert tilt_deg(h["quaternion"]).max() <= P.max_tilt_deg
    assert h["command"].min() >= 0 and h["command"].max() <= 1


def test_altitude_step_and_yaw_hold():
    h = fly(hold([0, 0, 6.0]), 10.0, position=(0, 0, 5))
    assert h["position"][-1, 2] == pytest.approx(6.0, abs=0.02)
    assert h["position"][:, 2].max() < 6.0 + 0.2             # modest overshoot


def test_recovers_from_attitude_disturbance():
    q0 = quat.from_euler(math.radians(20), math.radians(-15), 0.4)
    h = fly(hold([0, 0, 10]), 10.0, position=(0, 0, 10), quaternion=q0, body_rates=(1.0, -1.0, 0.5))
    roll, pitch, yaw = quat.to_euler(h["quaternion"][-1])
    assert (roll, pitch, yaw) == pytest.approx((0, 0, 0), abs=0.01)
    assert abs(h["position"][-1] - [0, 0, 10]).max() < 0.05
    assert h["position"][:, 2].min() > 8.0                    # didn't plunge during recovery


def test_follows_constant_velocity_line_with_feedforward():
    v = np.array([0.6, 0.0, 0.0])
    fn = lambda t: Setpoint(np.array([0, 0, 5]) + v * t, velocity=v)
    h = fly(fn, 12.0, position=(0, 0, 5), velocity=v)
    err = np.linalg.norm(h["position"][-1000:] - h["setpoint_position"][-1000:], axis=1)
    assert err.max() < 0.03


def test_tracks_smooth_curve_with_acceleration_feedforward():
    A, w = 0.5, 1.0
    fn = lambda t: Setpoint(np.array([0, A * math.sin(w * t), 5 + 0.3 * math.sin(0.5 * w * t)]),
                            velocity=np.array([0, A * w * math.cos(w * t), 0.3 * 0.5 * w * math.cos(0.5 * w * t)]),
                            acceleration=np.array([0, -A * w * w * math.sin(w * t), -0.3 * 0.25 * w * w * math.sin(0.5 * w * t)]))
    sp0 = fn(0.0)
    h = fly(fn, 15.0, position=sp0.position, velocity=sp0.velocity)
    err = np.linalg.norm(h["position"][2500:] - h["setpoint_position"][2500:], axis=1)
    assert err.max() < 0.05


def test_extreme_request_saturates_safely():
    """A target 50 m away demands more than the drone can do: limits must hold, no NaN, stays airborne."""
    h = fly(hold([50, 0, 5]), 6.0, position=(0, 0, 5))
    assert np.isfinite(h["position"]).all() and np.isfinite(h["command"]).all()
    assert h["command"].min() >= 0 and h["command"].max() <= 1
    assert tilt_deg(h["quaternion"]).max() <= P.max_tilt_deg + 3.0   # small overshoot of the tilt limit allowed
    assert np.linalg.norm(h["velocity"], axis=1).max() > 1.0         # actually moved
    assert h["position"][:, 2].min() > 3.0                           # kept altitude while tilting


def test_tilt_limit_binds_when_acceleration_limit_is_loose():
    """With the accel cap raised, the 20 deg tilt cap alone must hold the drone's attitude in check."""
    from dataclasses import replace
    p = replace(P, max_tilt_deg=20.0, max_accel_mps2=50.0)
    quad = Quadrotor(p)
    quad.reset(position=(0, 0, 5))
    h = simulate(quad, FlightController(p), hold([30, 0, 5]), 3.0)
    peak = tilt_deg(h["quaternion"]).max()
    assert 15.0 < peak <= 20.0 + 3.0     # reaches the cap (so it is binding) without blowing through it
