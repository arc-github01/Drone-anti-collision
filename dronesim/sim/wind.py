"""Wind model: a steady 3D wind plus optional Gauss-Markov turbulence.

World frame is ENU-like as everywhere else: +x = the drone's initial forward (approach) axis,
+y = left, +z = up.  Directions describe where the wind blows TOWARD:

    azimuth   psi in the horizontal plane, from +x toward +y (counter-clockwise seen from above)
    elevation eps from the horizontal plane toward +z

    w = V * ( cos(eps) cos(psi),  cos(eps) sin(psi),  sin(eps) )

so psi = 0 is a tailwind (pushes the drone forward), psi = 180 deg a headwind, psi = +90 deg blows toward
the drone's left, eps = +90 deg is a pure updraft.

Turbulence is one independent Ornstein-Uhlenbeck (first-order Gauss-Markov) process per axis,
    dX = -(X / tau) dt + sigma * sqrt(2 / tau) dW,
sampled with its EXACT discretisation  X[k+1] = a X[k] + sigma sqrt(1 - a^2) N(0,1),  a = exp(-dt/tau),
started from the stationary distribution N(0, sigma^2). Hence mean 0, variance sigma^2 and autocorrelation
exp(-|lag| / tau) hold exactly, independent of the sampling step. (This is the simple Gauss-Markov gust model,
not the full Dryden spectrum.) The realisation is fixed by `seed`, so runs are reproducible, and it is sampled on
its own fixed grid and linearly interpolated, so it does not depend on the physics step.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class WindConfig:
    speed_mps: float = 0.0
    azimuth_deg: float = 0.0        # direction the wind blows toward, in the horizontal plane (0 = +x, 90 = +y/left)
    elevation_deg: float = 0.0      # +90 = straight up
    turbulence_sigma_mps: float = 0.0   # per-axis standard deviation of the gust (m/s); 0 = steady wind
    turbulence_tau_s: float = 2.0       # gust correlation time (s)
    seed: int = 0

    def __post_init__(self):
        if self.speed_mps < 0 or self.turbulence_sigma_mps < 0:
            raise ValueError("wind speed and turbulence sigma must be >= 0")
        if self.turbulence_tau_s <= 0:
            raise ValueError("turbulence correlation time must be > 0")

    @property
    def active(self) -> bool:
        return self.speed_mps > 0 or self.turbulence_sigma_mps > 0

    def mean_vector(self) -> np.ndarray:
        return wind_vector(self.speed_mps, self.azimuth_deg, self.elevation_deg)

    def to_dict(self) -> dict:
        return asdict(self)


def wind_vector(speed: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    psi, eps = np.radians(azimuth_deg), np.radians(elevation_deg)
    return speed * np.array([np.cos(eps) * np.cos(psi), np.cos(eps) * np.sin(psi), np.sin(eps)])


def wind_angles(w) -> tuple:
    """Inverse of wind_vector: (speed, azimuth_deg, elevation_deg). Azimuth is 0 for a purely vertical wind."""
    w = np.asarray(w, dtype=float)
    v = float(np.linalg.norm(w))
    if v == 0.0:
        return 0.0, 0.0, 0.0
    return v, float(np.degrees(np.arctan2(w[1], w[0]))), float(np.degrees(np.arcsin(np.clip(w[2] / v, -1, 1))))


def components_in_approach_frame(w) -> dict:
    """Headwind is positive when the wind opposes forward flight (blows toward -x)."""
    w = np.asarray(w, dtype=float)
    return {"headwind_mps": float(-w[0]), "crosswind_left_mps": float(w[1]), "vertical_up_mps": float(w[2])}


class TurbulenceTrack:
    """Seeded OU gust series on a fixed grid, linearly interpolated (clamped at the ends)."""

    def __init__(self, sigma: float, tau: float, duration_s: float, seed: int = 0, dt: float = 0.01):
        self.dt = dt
        n = int(np.ceil(duration_s / dt)) + 2
        self.x = np.zeros((n, 3))
        if sigma > 0:
            rng = np.random.default_rng(seed)
            a = np.exp(-dt / tau)
            s = sigma * np.sqrt(1.0 - a * a)
            # Draw order matters: the start state first, then the noise block. The stream is sequential, so a longer
            # track is an extension of a shorter one (same seed => same gusts regardless of the flight duration).
            x = sigma * rng.standard_normal(3)          # stationary start
            eps = rng.standard_normal((n, 3))
            self.x[0] = x
            for k in range(1, n):
                x = a * x + s * eps[k]
                self.x[k] = x

    def at(self, t: float) -> np.ndarray:
        u = min(max(t, 0.0) / self.dt, len(self.x) - 1.000001)
        i = int(u)
        f = u - i
        return (1.0 - f) * self.x[i] + f * self.x[i + 1]


class WindModel:
    """Callable `wind(t, position) -> world-frame wind velocity (3,)`, the signature Quadrotor expects."""

    def __init__(self, cfg: Optional[WindConfig] = None, duration_s: float = 60.0):
        self.cfg = cfg or WindConfig()
        self.mean = self.cfg.mean_vector()
        self.turbulence = TurbulenceTrack(self.cfg.turbulence_sigma_mps, self.cfg.turbulence_tau_s,
                                          duration_s, self.cfg.seed)

    def __call__(self, t: float, position=None) -> np.ndarray:
        return self.mean + self.turbulence.at(t)
