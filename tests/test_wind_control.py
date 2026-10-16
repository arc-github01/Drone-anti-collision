import math

import numpy as np
import pytest

from dronesim.sim import quaternion as quat
from dronesim.sim.controller import FlightController, Gains, Setpoint
from dronesim.sim.loop import simulate
from dronesim.sim.params import GRAVITY, DroneParams
from dronesim.sim.quadrotor import Quadrotor
from dronesim.sim.wind import WindConfig, WindModel, wind_vector

P = DroneParams()
K = P.drag_coeff_kg_per_m / P.mass_kg
HOLD = np.array([0.0, 0.0, 5.0])


def fly(cfg, seconds=10.0, gains=None, setpoint=HOLD, start=HOLD, velocity=(0, 0, 0), sp_fn=None):
    quad = Quadrotor(P, wind_fn=WindModel(cfg, seconds + 1))
    quad.reset(position=start, velocity=velocity)
    ctrl = FlightController(P, gains)
    h = simulate(quad, ctrl, sp_fn or (lambda t: Setpoint(np.array(setpoint, float))), seconds)
    return h, ctrl


def tilt_deg(q):
    return math.degrees(math.acos(np.clip(quat.to_rotation_matrix(q)[2, 2], -1, 1)))


DIRECTIONS = [(5, 180, 0), (5, 0, 0), (5, 90, 0), (5, -90, 0), (4, 0, 90), (4, 0, -90),
              (5, 130, 35), (4, -40, -50), (6, 20, 70)]


@pytest.mark.parametrize("v,az,el", DIRECTIONS)
def test_steady_wind_in_any_3d_direction_is_rejected(v, az, el):
    """Hold position in steady wind: observer learns the drag acceleration, error returns to ~cm."""
    h, ctrl = fly(WindConfig(v, az, el))
    w = wind_vector(v, az, el)
    d_true = K * np.linalg.norm(w) * w                                   # drag acceleration on a stationary drone
    assert ctrl.disturbance_estimate == pytest.approx(d_true, abs=0.02 * np.linalg.norm(d_true) + 1e-3)
    assert np.linalg.norm(h["position"][-500:] - HOLD, axis=1).max() < 0.03
    assert h["saturated"].mean() == 0.0 and h["limited"][-500:].sum() == 0 and np.isfinite(h["command"]).all()


def test_steady_hover_tilt_matches_the_force_balance():
    """Rotor thrust must cancel gravity and drag:  F = m g e_z - m d  =>  tilt = atan(|d_h| / (g - d_z)),
    leaning INTO the wind (thrust's horizontal part opposes the drag, which points downwind)."""
    for v, az, el in [(5, 180, 0), (5, 130, 35), (6, 20, -30), (8, 90, 0)]:
        h, _ = fly(WindConfig(v, az, el), seconds=12)
        w = wind_vector(v, az, el)
        d = K * np.linalg.norm(w) * w
        f = np.array([0, 0, GRAVITY]) - d                                  # required specific force
        theory = math.degrees(math.atan2(np.linalg.norm(f[:2]), f[2]))
        assert tilt_deg(h["quaternion"][-1]) == pytest.approx(theory, abs=0.1)
        zb = quat.to_rotation_matrix(h["quaternion"][-1])[:, 2]            # body z in world = thrust direction
        assert zb == pytest.approx(f / np.linalg.norm(f), abs=2e-3)
        # total thrust = m |f|
        thrust = h["thrust"][-1].sum()
        assert thrust == pytest.approx(P.mass_kg * np.linalg.norm(f), rel=5e-3)


def test_without_the_observer_a_steady_wind_leaves_a_large_offset():
    h, _ = fly(WindConfig(5, 180, 0), gains=Gains(wind_compensation=False))
    err = np.linalg.norm(h["position"][-500:] - HOLD, axis=1).max()
    assert err > 0.3                                                       # ~ d / Kp = 1.875 / 4 = 0.47 m
    assert err == pytest.approx(1.875 / Gains().kp_pos[0], rel=0.1)


def test_observer_is_silent_in_still_air_and_learns_speed_drag():
    h, ctrl = fly(WindConfig(), seconds=6)
    assert np.abs(ctrl.disturbance_estimate).max() < 1e-6
    # constant 3 m/s forward flight in still air: drag acceleration = -K |v| v
    v = np.array([3.0, 0.0, 0.0])
    h, ctrl = fly(WindConfig(), seconds=10, start=HOLD, velocity=v,
                  sp_fn=lambda t: Setpoint(HOLD + v * t, velocity=v))
    assert ctrl.disturbance_estimate == pytest.approx([-K * 9.0, 0, 0], abs=0.02)


def test_wind_beyond_the_drones_authority_saturates_safely():
    """12 m/s: needed lateral accel K*144 = 10.8 m/s^2 > 6 m/s^2 limit (tilt 47.8 deg > 35 deg): cannot hold station."""
    need = K * 144.0
    assert need > P.max_accel_mps2 and math.degrees(math.atan2(need, GRAVITY)) > P.max_tilt_deg
    h, _ = fly(WindConfig(12, 0, 0), seconds=12)                           # tailwind: pushed downwind
    assert np.isfinite(h["position"]).all() and np.isfinite(h["command"]).all()
    assert h["limited"].mean() > 0.5                                        # the tilt/acceleration limit is what binds
    assert h["command"].min() >= 0 and h["command"].max() <= 1
    assert np.linalg.norm(h["position"][-1] - HOLD) > 1.0                   # genuinely loses station
    assert max(tilt_deg(q) for q in h["quaternion"]) <= P.max_tilt_deg + 3.0
    assert h["position"][:, 2].min() > 3.0                                  # but does not fall out of the sky


def test_gusty_wind_is_tracked_to_centimetres_and_reproducibly():
    cfg = WindConfig(speed_mps=3, azimuth_deg=90, elevation_deg=0, turbulence_sigma_mps=1.0, turbulence_tau_s=2.0, seed=0)
    h, _ = fly(cfg, seconds=20)
    err = np.linalg.norm(h["position"][1000:] - HOLD, axis=1)
    assert np.sqrt((err**2).mean()) < 0.10 and err.max() < 0.25
    assert h["saturated"].mean() == 0.0
    # the recorded wind is mean + gust, with the gust's standard deviation near sigma over a long record
    gust = h["wind"] - wind_vector(3, 90, 0)
    assert np.abs(gust).max() > 0.5
    h2, _ = fly(cfg, seconds=20)
    assert np.array_equal(h["position"], h2["position"]) and np.array_equal(h["wind"], h2["wind"])


def test_recorded_wind_equals_the_model():
    cfg = WindConfig(4, 60, 20, 0.5, 1.5, 7)
    h, _ = fly(cfg, seconds=3)
    wm = WindModel(cfg, 4.0)
    for i in (0, 100, 700, 1400):
        assert h["wind"][i] == pytest.approx(wm(h["t"][i], None))
