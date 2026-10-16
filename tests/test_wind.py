import math
from dataclasses import replace

import numpy as np
import pytest

from dronesim.sim.params import GRAVITY, DroneParams
from dronesim.sim.quadrotor import Quadrotor
from dronesim.sim.wind import (TurbulenceTrack, WindConfig, WindModel, components_in_approach_frame, wind_angles,
                               wind_vector)

P = DroneParams()
K = P.drag_coeff_kg_per_m / P.mass_kg          # drag acceleration coefficient c_d / m


# ------------------------------------------------------------------ direction maths
def test_axis_directions_and_speed():
    assert wind_vector(5, 0, 0) == pytest.approx([5, 0, 0])          # tailwind
    assert wind_vector(5, 180, 0) == pytest.approx([-5, 0, 0], abs=1e-12)   # headwind
    assert wind_vector(5, 90, 0) == pytest.approx([0, 5, 0], abs=1e-12)     # toward the drone's left
    assert wind_vector(5, -90, 0) == pytest.approx([0, -5, 0], abs=1e-12)   # toward the right
    assert wind_vector(5, 0, 90) == pytest.approx([0, 0, 5], abs=1e-12)     # updraft
    assert wind_vector(5, 123, -90) == pytest.approx([0, 0, -5], abs=1e-12) # downdraft, azimuth irrelevant


def test_general_direction_has_correct_magnitude_and_angles():
    for v, az, el in [(3.0, 30, 20), (7.5, -120, -45), (1.2, 179, 80), (10, 0.5, 0.25)]:
        w = wind_vector(v, az, el)
        assert np.linalg.norm(w) == pytest.approx(v)
        assert wind_angles(w) == pytest.approx((v, az, el), abs=1e-9)
        # horizontal part has magnitude v cos(el), vertical part v sin(el)
        assert math.hypot(w[0], w[1]) == pytest.approx(v * math.cos(math.radians(el)))
        assert w[2] == pytest.approx(v * math.sin(math.radians(el)))
    assert wind_angles([0, 0, 0]) == (0.0, 0.0, 0.0)


def test_approach_frame_components():
    c = components_in_approach_frame(wind_vector(6, 180, 0))
    assert c["headwind_mps"] == pytest.approx(6) and abs(c["crosswind_left_mps"]) < 1e-12
    c = components_in_approach_frame(wind_vector(4, 90, 30))
    assert c["crosswind_left_mps"] == pytest.approx(4 * math.cos(math.radians(30)))
    assert c["vertical_up_mps"] == pytest.approx(2.0) and abs(c["headwind_mps"]) < 1e-12


def test_config_validation():
    for bad in (dict(speed_mps=-1), dict(turbulence_sigma_mps=-0.1), dict(turbulence_tau_s=0)):
        with pytest.raises(ValueError):
            WindConfig(**bad)
    assert not WindConfig().active and WindConfig(speed_mps=1).active and WindConfig(turbulence_sigma_mps=1).active


# ------------------------------------------------------------------ Ornstein-Uhlenbeck turbulence
def _series(sigma, tau, dt, n, seed):
    tr = TurbulenceTrack(sigma, tau, (n - 2) * dt, seed, dt=dt)
    return tr.x


def test_turbulence_statistics_match_the_ou_process():
    sigma, tau, dt = 1.3, 2.0, 0.02
    x = _series(sigma, tau, dt, 400_000, seed=3)[:, 0]
    assert abs(x.mean()) < 0.05
    assert x.var() == pytest.approx(sigma**2, rel=0.03)
    for lag_s in (0.5, 2.0, 4.0):                       # autocorrelation = exp(-lag/tau)
        k = int(round(lag_s / dt))
        r = np.mean(x[:-k] * x[k:]) / x.var()
        assert r == pytest.approx(math.exp(-lag_s / tau), abs=0.03)


def test_exact_discretisation_is_step_size_independent():
    """Exact discretisation: variance and autocorrelation do not depend on dt. The tolerance is 4 standard errors
    of the estimators for an AR(1) series of that length, so the test is as tight as the statistics allow."""
    sigma, tau, total_s = 0.8, 1.5, 6000.0
    for dt in (0.005, 0.05, 0.3):
        n = int(total_s / dt) + 2
        x = _series(sigma, tau, dt, n, seed=5)[:, 1]
        a = math.exp(-dt / tau)
        se_var = math.sqrt(2 * (1 + a * a) / ((1 - a * a) * n))               # relative std of the sample variance
        se_rho = math.sqrt((1 + a * a) / ((1 - a * a) * n)) * 1.5              # generous bound for the lag-tau autocorrelation
        k = int(round(tau / dt))
        assert x.var() == pytest.approx(sigma**2, rel=4 * se_var)
        assert np.mean(x[:-k] * x[k:]) / x.var() == pytest.approx(math.exp(-1), abs=4 * se_rho)


def test_axes_are_independent_and_seed_controls_the_realisation():
    x = _series(1.0, 1.0, 0.02, 200_000, seed=11)
    c = np.corrcoef(x.T)
    assert abs(c[0, 1]) < 0.03 and abs(c[0, 2]) < 0.03 and abs(c[1, 2]) < 0.03
    assert np.array_equal(x, _series(1.0, 1.0, 0.02, 200_000, seed=11))
    assert not np.array_equal(x, _series(1.0, 1.0, 0.02, 200_000, seed=12))


def test_gust_realisation_does_not_depend_on_the_track_length():
    """Same seed => same gusts whatever the flight duration (a longer track extends a shorter one)."""
    short = TurbulenceTrack(1.0, 2.0, 5.0, seed=9)
    long = TurbulenceTrack(1.0, 2.0, 50.0, seed=9)
    assert np.array_equal(short.x, long.x[: len(short.x)])
    for t in (0.0, 1.234, 4.9):
        assert short.at(t) == pytest.approx(long.at(t), abs=1e-15)


def test_turbulence_is_continuous_clamped_and_zero_when_off():
    tr = TurbulenceTrack(1.0, 2.0, 5.0, seed=1)
    assert tr.at(-1.0) == pytest.approx(tr.x[0]) and np.isfinite(tr.at(1e6)).all()
    ts = np.linspace(0, 5, 2001)
    v = np.array([tr.at(t) for t in ts])
    assert np.abs(np.diff(v, axis=0)).max() < 0.2                # piecewise-linear: no jumps
    assert np.array_equal(TurbulenceTrack(0.0, 2.0, 5.0).at(2.3), np.zeros(3))


def test_wind_model_is_mean_plus_gust():
    m = WindModel(WindConfig(speed_mps=4, azimuth_deg=60, elevation_deg=10, turbulence_sigma_mps=0.7, seed=2), 10.0)
    for t in (0.0, 1.234, 9.9):
        assert m(t, None) == pytest.approx(wind_vector(4, 60, 10) + m.turbulence.at(t))
    steady = WindModel(WindConfig(speed_mps=3, azimuth_deg=0), 10.0)
    assert steady(2.0, np.zeros(3)) == pytest.approx([3, 0, 0])


# ------------------------------------------------------------------ drag in wind (physics)
def _hover_accel(vel, wind, quat=(1, 0, 0, 0)):
    """World acceleration of a drone whose rotors exactly support its weight, in the given wind."""
    q = Quadrotor(P, wind_fn=lambda t, p: np.asarray(wind, dtype=float))
    q.reset(velocity=vel, quaternion=quat)
    return q.step(np.full(4, P.hover_command)).accel_world


@pytest.mark.parametrize("w", [(5, 0, 0), (-4, 0, 0), (0, 3, 0), (0, 0, 6), (2, -3, 4), (-1.5, 2.5, -3.5)])
def test_stationary_drone_in_wind_feels_c_d_abs_w_w(w):
    """v = 0  =>  v_rel = -w  =>  a = -(c_d/m)|v_rel| v_rel = +(c_d/m)|w| w, pointing downwind (any 3D direction)."""
    w = np.array(w, dtype=float)
    assert _hover_accel((0, 0, 0), w) == pytest.approx(K * np.linalg.norm(w) * w, rel=1e-9)


def test_drone_moving_with_the_wind_feels_no_drag():
    for w in ([4, 1, -2], [0, 0, 3], [-5, 2, 1]):
        assert _hover_accel(w, w) == pytest.approx(0, abs=1e-12)


def test_drag_depends_on_relative_velocity_only():
    base = _hover_accel((3, 0, 0), (1, 0, 0))                      # v_rel = (2,0,0)
    assert base == pytest.approx([-K * 4, 0, 0], rel=1e-9)
    assert _hover_accel((13, 0, 0), (11, 0, 0)) == pytest.approx(base, rel=1e-9)   # same v_rel, other frame
    # quadratic law: doubling the relative speed quadruples the force
    a1, a2 = _hover_accel((0, 0, 0), (2, 0, 0))[0], _hover_accel((0, 0, 0), (4, 0, 0))[0]
    assert a2 == pytest.approx(4 * a1, rel=1e-9)


def test_unpowered_fall_in_horizontal_wind_converges_to_wind_and_terminal_velocity():
    """Horizontal force balance -c|v_rel|(v_x - w_x) = 0  =>  v_x -> w_x ; vertical: |v_z| -> sqrt(m g / c)."""
    wind = np.array([3.0, -2.0, 0.0])
    q = Quadrotor(P, dt=0.005, wind_fn=lambda t, p: wind)
    q.reset(position=(0, 0, 1e5), rotor_speeds=np.zeros(4))
    for _ in range(int(60 / 0.005)):
        q.step(np.zeros(4))
    assert q.velocity[:2] == pytest.approx(wind[:2], abs=0.02)
    assert -q.velocity[2] == pytest.approx(math.sqrt(P.mass_kg * GRAVITY / P.drag_coeff_kg_per_m), rel=0.01)


def test_integration_with_wind_converges_with_step_size():
    """Gusty wind, short steps vs long steps: same trajectory (turbulence lives on its own fixed grid)."""
    def run(dt):
        wm = WindModel(WindConfig(speed_mps=3, azimuth_deg=40, elevation_deg=15, turbulence_sigma_mps=1.0, seed=4), 6.0)
        q = Quadrotor(P, dt=dt, wind_fn=wm)
        q.reset(velocity=(1, 0, 0))
        for _ in range(int(round(4.0 / dt))):
            q.step(np.full(4, P.hover_command))
        return q.position, q.velocity
    (p1, v1), (p2, v2) = run(0.004), run(0.001)
    assert p1 == pytest.approx(p2, abs=2e-4) and v1 == pytest.approx(v2, abs=2e-4)


def test_output_reports_the_wind_used_and_default_is_still_zero():
    w = np.array([1.0, 2.0, 3.0])
    q = Quadrotor(P, wind_fn=lambda t, p: w)
    q.reset()
    assert q.step(np.full(4, 0.7)).wind_world == pytest.approx(w)
    q0 = Quadrotor(P)
    q0.reset()
    assert q0.step(np.full(4, 0.7)).wind_world == pytest.approx(0)
