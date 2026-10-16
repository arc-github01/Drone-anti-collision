import math
from dataclasses import replace

import numpy as np
import pytest

from dronesim.sim import motors, quaternion as quat
from dronesim.sim.params import GRAVITY, DroneParams
from dronesim.sim.quadrotor import Quadrotor

P = DroneParams()


def run(q: Quadrotor, u, seconds):
    out = None
    for _ in range(int(round(seconds / q.dt))):
        out = q.step(u)
    return out


def hover_u():
    return np.full(4, P.hover_command)


# ---------------------------------------------------------------- quaternion
def test_euler_roundtrip_and_rotation():
    for r, p, y in [(0.1, -0.2, 0.3), (-0.5, 0.4, 2.0), (0, 0, 0)]:
        q = quat.from_euler(r, p, y)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert quat.to_euler(q) == pytest.approx((r, p, y), abs=1e-12)
    # yaw +90 deg takes body +x (forward) to world +y
    R = quat.to_rotation_matrix(quat.from_euler(0, 0, math.pi / 2))
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-12)


# ---------------------------------------------------------------- motor model
def test_thrust_is_kT_omega_squared_and_quantities_are_distinct():
    om = np.array([300.0, 600.0, 900.0, 1500.0])
    assert motors.thrust(om, P) == pytest.approx(P.k_thrust * om**2)
    assert motors.thrust(np.array([P.omega_max_radps]), P)[0] == pytest.approx(P.max_thrust_per_motor_n)
    # command != omega != thrust != power
    u = np.array([0.5])
    w = motors.command_to_omega(u, P)
    T, Pw = motors.thrust(w, P), motors.electrical_power(w, P)
    assert len({float(u[0]), float(w[0]), float(T[0]), float(Pw[0])}) == 4
    assert Pw[0] == pytest.approx(P.k_torque * w[0] ** 3 / P.motor_efficiency + P.motor_idle_power_w)
    assert motors.electrical_power(np.array([800.0]), P) > motors.electrical_power(np.array([400.0]), P)


def test_command_is_saturated_to_unit_interval():
    q = Quadrotor(P)
    q.reset()
    run(q, [5.0, 5.0, -3.0, 5.0], 1.0)
    assert q.rotor_speeds.max() <= P.omega_max_radps + 1e-6
    assert q.rotor_speeds.min() >= 0.0
    assert q.rotor_speeds[2] == pytest.approx(0.0, abs=1.0)        # driven toward 0, not negative
    assert q.rotor_speeds[0] == pytest.approx(P.omega_max_radps, rel=1e-6)


def test_motor_first_order_lag_time_constant():
    q = Quadrotor(P, dt=0.0005)
    q.reset(rotor_speeds=np.zeros(4))
    run(q, np.ones(4), P.motor_time_constant_s)
    assert q.rotor_speeds[0] / P.omega_max_radps == pytest.approx(1 - math.exp(-1), rel=1e-3)


def test_mixer_roundtrip_and_hover_equilibrium():
    A = motors.mixer_matrix(P)
    wrench = np.array([1.0, 0.003, -0.002, 0.001])
    assert A @ motors.thrusts_from_wrench(wrench, P) == pytest.approx(wrench)
    T = motors.thrusts_from_wrench([P.weight_n, 0, 0, 0], P)
    assert T == pytest.approx(np.full(4, P.weight_n / 4))
    assert motors.thrust_to_command(T, P) == pytest.approx(np.full(4, P.hover_command))


# ---------------------------------------------------------------- translation
def test_hover_holds_position_and_attitude():
    q = Quadrotor(P)
    q.reset(position=(1, 2, 3))
    run(q, hover_u(), 10.0)
    assert q.position == pytest.approx([1, 2, 3], abs=1e-6)
    assert q.velocity == pytest.approx(0, abs=1e-6)
    assert q.quaternion == pytest.approx([1, 0, 0, 0], abs=1e-9)


def test_free_fall_matches_analytic_without_drag():
    q = Quadrotor(replace(P, drag_coeff_kg_per_m=0.0))
    q.reset(position=(0, 0, 100), rotor_speeds=np.zeros(4))
    run(q, np.zeros(4), 2.0)
    assert q.position[2] == pytest.approx(100 - 0.5 * GRAVITY * 4.0, rel=1e-9)
    assert q.velocity[2] == pytest.approx(-GRAVITY * 2.0, rel=1e-9)


def test_terminal_velocity_with_drag():
    q = Quadrotor(P, dt=0.005)
    q.reset(position=(0, 0, 1e4), rotor_speeds=np.zeros(4))
    run(q, np.zeros(4), 40.0)
    v_term = math.sqrt(P.mass_kg * GRAVITY / P.drag_coeff_kg_per_m)
    assert -q.velocity[2] == pytest.approx(v_term, rel=0.01)


def test_tilted_hover_thrust_accelerates_horizontally():
    """Pitch about +y by theta (nose down): thrust T=mg/cos(theta) gives a_x = g tan(theta), a_z = 0."""
    theta = math.radians(10)
    q = Quadrotor(replace(P, drag_coeff_kg_per_m=0.0))
    T_each = P.weight_n / math.cos(theta) / 4
    om = np.sqrt(T_each / P.k_thrust)
    q.reset(quaternion=quat.from_euler(0, theta, 0), rotor_speeds=np.full(4, om))
    out = q.step(np.full(4, om / P.omega_max_radps))
    assert out.accel_world[0] == pytest.approx(GRAVITY * math.tan(theta), rel=1e-9)
    assert out.accel_world[1] == pytest.approx(0, abs=1e-9)
    assert out.accel_world[2] == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------- torques and attitude
def _rates_after(u, seconds=0.05):
    q = Quadrotor(P)
    q.reset()
    run(q, u, seconds)
    return q.body_rates


def test_differential_thrust_torque_signs():
    h = P.hover_command
    d = 0.1
    # Motors: 0 FR(ccw), 1 RR(cw), 2 RL(ccw), 3 FL(cw)
    front_up = _rates_after(np.array([h + d, h - d, h - d, h + d]))
    assert front_up[1] < 0 and abs(front_up[0]) < 1e-6 and abs(front_up[2]) < 1e-6   # nose up = -omega_y
    right_up = _rates_after(np.array([h + d, h + d, h - d, h - d]))
    assert right_up[0] < 0 and abs(right_up[1]) < 1e-6 and abs(right_up[2]) < 1e-6   # right side up = -omega_x
    ccw_faster = _rates_after(np.array([h + d, h - d, h + d, h - d]))
    assert ccw_faster[2] < 0 and abs(ccw_faster[0]) < 1e-6 and abs(ccw_faster[1]) < 1e-6  # body reacts opposite
    assert _rates_after(hover_u()) == pytest.approx(0, abs=1e-9)


def test_wrench_to_attitude_acceleration_matches_inertia():
    """Commanding tau_x = 1e-3 N m for a short time gives omega_x ~ tau/Ixx * t (before lag shapes it)."""
    T = motors.thrusts_from_wrench([P.weight_n, 1e-3, 0, 0], P)
    q = Quadrotor(replace(P, angular_damping_nms=0.0))
    q.reset(rotor_speeds=motors.command_to_omega(motors.thrust_to_command(T, P), P))
    run(q, motors.thrust_to_command(T, P), 0.1)
    assert q.body_rates[0] == pytest.approx(1e-3 / P.inertia_kgm2[0] * 0.1, rel=1e-3)
    assert abs(q.body_rates[1]) < 1e-6


def test_torque_free_rotation_conserves_angular_momentum_and_energy():
    p = replace(P, inertia_kgm2=(1e-4, 2e-4, 3e-4), angular_damping_nms=0.0, drag_coeff_kg_per_m=0.0,
                rotor_inertia_kgm2=0.0)
    q = Quadrotor(p, dt=0.002)
    q.reset(position=(0, 0, 1e3), body_rates=(5.0, 1.0, 3.0), rotor_speeds=np.zeros(4))
    J = p.inertia_matrix

    def L_world():
        R = quat.to_rotation_matrix(q.quaternion)
        return R @ (J @ q.body_rates)

    L0, E0 = L_world(), 0.5 * q.body_rates @ J @ q.body_rates
    run(q, np.zeros(4), 5.0)
    assert L_world() == pytest.approx(L0, rel=1e-6)
    assert 0.5 * q.body_rates @ J @ q.body_rates == pytest.approx(E0, rel=1e-6)
    assert np.linalg.norm(q.quaternion) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- integration properties
def test_time_is_exact_multiple_of_fixed_dt():
    q = Quadrotor(P, dt=0.002)
    q.reset()
    run(q, hover_u(), 1.0)
    assert q.n_steps == 500 and q.t == pytest.approx(1.0, abs=1e-12)


def test_step_size_convergence():
    def maneuver(dt):
        q = Quadrotor(P, dt=dt)
        q.reset()
        h, d = P.hover_command, 0.05
        run(q, [h + d, h + d, h - d, h - d], 0.3)
        run(q, hover_u(), 0.7)
        return q.position, q.quaternion
    p1, q1 = maneuver(0.004)
    p2, q2 = maneuver(0.001)
    assert p1 == pytest.approx(p2, abs=1e-5)
    assert q1 == pytest.approx(q2, abs=1e-5)


def test_default_wind_is_zero_and_deterministic():
    a, b = Quadrotor(P), Quadrotor(P)
    for q in (a, b):
        q.reset(velocity=(1, 0, 0))
        run(q, hover_u() * 1.01, 1.0)
    assert np.array_equal(a.state, b.state)
