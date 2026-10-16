"""Quadrotor rigid-body dynamics with four motors, integrated with fixed-step RK4.

State (17): p(3) world position, v(3) world velocity, q(4) body->world quaternion [w,x,y,z],
w(3) body angular velocity, Om(4) rotor angular speeds.  World ENU (z up); body FLU.
The command u (4, in [0,1]) is held constant over each step (zero-order hold).

The only way to move the drone is `step(u)`: nothing here is animated or scripted.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from dronesim.sim import motors, quaternion as quat
from dronesim.sim.params import GRAVITY, DroneParams

P, V, Q, W, OM = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 17)

# wind_fn(t, position) -> wind velocity in the world frame (m/s), e.g. dronesim.sim.wind.WindModel.
# Drag uses the AIR-RELATIVE velocity v - wind:  F = -c_d |v - w| (v - w).  Default: still air.
# It is called with the same t for the RK4 mid-stages, so it must be a deterministic function of t.
WindFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class Outputs:
    """Everything worth logging for one instant (all separate physical quantities)."""
    command: np.ndarray        # u, [0,1]
    omega: np.ndarray          # rad/s
    thrust: np.ndarray         # N per motor
    power_w: np.ndarray        # electrical power per motor
    torque_body: np.ndarray    # N m (x, y, z) applied to the body by the rotors
    accel_world: np.ndarray    # m/s^2 (linear acceleration, includes gravity and drag)
    wind_world: np.ndarray     # m/s, wind velocity the drag law used at the start of the step


class Quadrotor:
    def __init__(self, params: Optional[DroneParams] = None, dt: float = 0.002,
                 wind_fn: Optional[WindFn] = None):
        self.p = params or DroneParams()
        self.dt = dt
        self.wind_fn = wind_fn
        self.n_steps = 0
        self.state = np.zeros(17)
        self.state[Q] = [1.0, 0.0, 0.0, 0.0]
        self._J = self.p.inertia_matrix
        self._Jinv = np.linalg.inv(self._J)

    # -- time / state access -------------------------------------------------
    @property
    def t(self) -> float:
        return self.n_steps * self.dt   # exact multiple of dt; independent of wall clock

    @property
    def position(self): return self.state[P].copy()
    @property
    def velocity(self): return self.state[V].copy()
    @property
    def quaternion(self): return self.state[Q].copy()
    @property
    def body_rates(self): return self.state[W].copy()
    @property
    def rotor_speeds(self): return self.state[OM].copy()

    def reset(self, position=(0, 0, 0), velocity=(0, 0, 0), quaternion=(1, 0, 0, 0),
              body_rates=(0, 0, 0), rotor_speeds=None):
        """Rotor speeds default to the hover speed (drone starts airborne and in equilibrium)."""
        self.n_steps = 0
        s = self.state
        s[P], s[V], s[W] = position, velocity, body_rates
        s[Q] = quat.normalize(np.asarray(quaternion, dtype=float))
        s[OM] = (motors.command_to_omega(np.full(4, self.p.hover_command), self.p)
                 if rotor_speeds is None else rotor_speeds)

    # -- dynamics --------------------------------------------------------------
    def _wind(self, t, pos):
        return np.zeros(3) if self.wind_fn is None else np.asarray(self.wind_fn(t, pos), dtype=float)

    def _deriv(self, t: float, x: np.ndarray, omega_cmd: np.ndarray):
        p = self.p
        om = x[OM]
        T = motors.thrust(om, p)
        om_dot = (omega_cmd - om) / p.motor_time_constant_s

        xy = p.motor_positions
        tau = np.array([
            np.dot(xy[:, 1], T),                              # roll:  sum(y_i T_i)
            -np.dot(xy[:, 0], T),                             # pitch: -sum(x_i T_i)
            -np.sum(p.spin_dir * (motors.drag_torque(om, p) + p.rotor_inertia_kgm2 * om_dot)),
        ])

        R = quat.to_rotation_matrix(x[Q])
        v_rel = x[V] - self._wind(t, x[P])
        force = R @ np.array([0.0, 0.0, T.sum()]) + np.array([0.0, 0.0, -p.mass_kg * GRAVITY]) \
            - p.drag_coeff_kg_per_m * np.linalg.norm(v_rel) * v_rel
        accel = force / p.mass_kg

        w = x[W]
        w_dot = self._Jinv @ (tau - np.cross(w, self._J @ w) - p.angular_damping_nms * w)
        q_dot = 0.5 * quat.multiply(x[Q], np.array([0.0, *w]))

        dx = np.empty(17)
        dx[P], dx[V], dx[Q], dx[W], dx[OM] = x[V], accel, q_dot, w_dot, om_dot
        return dx, tau, accel, T

    def step(self, u) -> Outputs:
        """Advance one fixed physics step with motor commands u (4 values, clipped to [0,1])."""
        u = motors.clip_command(u)
        om_cmd = motors.command_to_omega(u, self.p)
        t, h, x = self.t, self.dt, self.state
        k1, tau, accel, T = self._deriv(t, x, om_cmd)       # outputs reported at the step start
        k2 = self._deriv(t + h / 2, x + h / 2 * k1, om_cmd)[0]
        k3 = self._deriv(t + h / 2, x + h / 2 * k2, om_cmd)[0]
        k4 = self._deriv(t + h, x + h * k3, om_cmd)[0]
        out = Outputs(u.copy(), x[OM].copy(), T.copy(), motors.electrical_power(x[OM], self.p),
                      tau.copy(), accel.copy(), self._wind(t, x[P]))
        self.state = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state[Q] = quat.normalize(self.state[Q])
        self.n_steps += 1
        return out
