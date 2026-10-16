"""Collision geometry: axis-aligned box obstacles, a spherical drone, an optional floor.

Clearance = (distance from the drone centre to the obstacle surface) - drone_radius.
Negative clearance means the drone body intersects the obstacle.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Box:
    center: np.ndarray
    half: np.ndarray        # half extents along world x, y, z

    @classmethod
    def from_obstacle(cls, ob: dict, z_offset: float = 0.0) -> "Box":
        """Build from a scenario obstacle (thin panel facing -x: thickness along x)."""
        c = np.array(ob["center_m"], dtype=float) + [0.0, 0.0, z_offset]
        half = 0.5 * np.array([ob["thickness_m"], ob["width_m"], ob["height_m"]], dtype=float)
        return cls(c, half)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance from points (N,3) to the box surface: >0 outside, <0 inside."""
        d = np.abs(np.atleast_2d(pts) - self.center) - self.half
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(d.max(axis=1), 0.0)
        return outside + inside


@dataclass
class Environment:
    boxes: List[Box]
    drone_radius_m: float = 0.10       # assumed (Tello-sized airframe + guards)
    safety_margin_m: float = 0.25      # required clearance beyond the drone body
    floor_z_m: Optional[float] = 0.0   # None disables the floor

    def obstacle_clearance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        d = np.min([b.signed_distance(pts) for b in self.boxes], axis=0)
        return d - self.drone_radius_m

    def floor_clearance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        if self.floor_z_m is None:
            return np.full(len(pts), np.inf)
        return pts[:, 2] - self.floor_z_m - self.drone_radius_m

    def clearance(self, pts: np.ndarray) -> np.ndarray:
        return np.minimum(self.obstacle_clearance(pts), self.floor_clearance(pts))


def predict_collision(env: Environment, times: np.ndarray, pts: np.ndarray) -> dict:
    """Check a sampled trajectory. `collides`: body touches an obstacle. `violates_margin`:
    clearance drops below the safety margin. Times are of the first such event."""
    c = env.clearance(pts)
    i_min = int(np.argmin(c))
    hit = np.where(c < 0.0)[0]
    viol = np.where(c < env.safety_margin_m)[0]
    return {
        "collides": bool(len(hit)),
        "violates_margin": bool(len(viol)),
        "t_first_contact_s": float(times[hit[0]]) if len(hit) else None,
        "t_first_violation_s": float(times[viol[0]]) if len(viol) else None,
        "min_clearance_m": float(c[i_min]),
        "t_min_clearance_s": float(times[i_min]),
    }
