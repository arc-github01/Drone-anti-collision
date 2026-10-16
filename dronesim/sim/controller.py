"""Cascaded flight controller: trajectory setpoint -> four motor commands.

    position/velocity (PD + I + accel feed-forward)  -> desired force vector (tilt/accel limited)
    geometric attitude control (rotation matrices)   -> desired body rates
    body-rate P loop + gyroscopic cancellation       -> body torques
    [collective thrust, torques] --mixer + desaturation--> per-motor thrust -> command in [0,1]

The controller only ever outputs motor commands; it never touches the simulated state.
It reads the (perfect, noise-free) state: there is no sensor noise model.

Disturbance rejection (wind, drag): a Luenberger observer on the velocity estimates the unmodelled
acceleration d (aerodynamic drag in the moving air, including any wind) from the rotors' own thrust:

    true:      v' = (R e_z T)/m - g e_z + d                    (d = drag acceleration)
    observer:  vh' = a_thrust + dh + 2 w0 (v - vh),   dh' = w0^2 (v - vh),     a_thrust = (R e_z T)/m - g e_z

The error dynamics are  e'' + 2 w0 e' + w0^2 e = d'  (critically damped, bandwidth w0), so dh -> d for a
constant d with no steady-state error; the commanded acceleration is reduced by dh. T is the MEASURED total
thrust (rotor speeds, as ESC telemetry would give); the wind itself is never read by the controller.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dronesim.sim import motors, quaternion as quat
from dronesim.sim.params import GRAVITY, DroneParams


@dataclass
class Gains:
    kp_pos: tuple = (4.0, 4.0, 8.0)       # 1/s^2
    kd_vel: tuple = (3.5, 3.5, 5.0)       # 1/s
    ki_pos: tuple = (0.5, 0.5, 1.0)       # 1/s^3 (integral of position error)
    i_limit_mps2: float = 1.5             # clamp on the integral acceleration term
    i_active_radius_m: float = 0.3        # integrate only this close to the target (avoids step windup)
    k_att: tuple = (7.0, 7.0, 3.0)        # rad/s per rad of attitude error (roll, pitch, yaw)
    k_rate: tuple = (18.0, 18.0, 10.0)    # rad/s^2 per rad/s of rate error
    max_rate_radps: float = 8.0
    wind_compensation: bool = True        # disturbance observer on/off
    obs_bandwidth_radps: float = 5.0      # w0: observer convergence ~ 4/w0 seconds
    # With the observer cancelling constant disturbances, the integral term is only a slow residual trim. At full
    # strength it winds up during transients and then unwinds at ~ki/kp (an ~8 s tail), so it is scaled down.
    observer_integral_scale: float = 0.05


@dataclass
class Setpoint:
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


@dataclass
class ControlOutput:
    command: np.ndarray            # 4 motor commands in [0,1]  <- what the graphs show
    thrust_total_n: float          # commanded collective thrust after desaturation
    torque_nm: np.ndarray          # commanded body torque after desaturation
    accel_cmd: np.ndarray          # commanded world acceleration (before tilt limiting)
    q_des: np.ndarray              # desired attitude quaternion
    saturated: bool                # True if the mixer had to cut yaw/attitude/thrust to fit limits
    limited: bool = False          # True if the acceleration or tilt limit clipped the demanded force vector


def desaturate(thrust_total: float, torque: np.ndarray, p: DroneParams):
    """Fit [T, tau] into 0 <= T_i <= Tmax per motor. Priority: roll/pitch > thrust > yaw.

    T_i = T_att_i + c, where T_att (zero-sum) realises the torques and c is the common-mode thrust.
    Returns (per-motor thrust, thrust_total_used, torque_used, saturated).
    """
    tmax = p.max_thrust_per_motor_n
    tau = np.asarray(torque, dtype=float).copy()
    sat = False

    def att(tq):
        return motors.thrusts_from_wrench([0.0, *tq], p)

    def span(tq):
        a = att(tq)
        return a.max() - a.min()

    if span(tau) > tmax:                        # 1) sacrifice yaw first (bisection on its scale)
        sat = True
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if span(np.array([tau[0], tau[1], tau[2] * mid])) <= tmax:
                lo = mid
            else:
                hi = mid
        tau[2] *= lo
        if span(tau) > tmax:                    # 2) then scale roll/pitch uniformly
            s = tmax / span(tau)
            tau[:2] *= s
    a = att(tau)
    c_lo, c_hi = -a.min(), tmax - a.max()       # feasible common-mode thrust range per motor
    c = np.clip(thrust_total / 4.0, c_lo, c_hi)
    if abs(c - thrust_total / 4.0) > 1e-12:
        sat = True
    return a + c, 4.0 * c, tau, sat


class FlightController:
    def __init__(self, params: Optional[DroneParams] = None, gains: Optional[Gains] = None):
        self.p = params or DroneParams()
        self.g = gains or Gains()
        self._J = self.p.inertia_matrix
        self._i_term = np.zeros(3)
        self._last_saturated = False
        self._vhat = None            # observer velocity estimate
        self._dhat = np.zeros(3)     # observer disturbance-acceleration estimate (world frame)
        self._last_thrust = self.p.weight_n

    def reset(self):
        self._i_term[:] = 0.0
        self._last_saturated = False
        self._vhat, self._dhat = None, np.zeros(3)
        self._last_thrust = self.p.weight_n

    @property
    def disturbance_estimate(self) -> np.ndarray:
        """Estimated disturbance (drag/wind) acceleration in the world frame, m/s^2."""
        return self._dhat.copy()

    def update(self, position, velocity, quaternion, body_rates, sp: Setpoint, dt: float,
               measured_thrust_n: Optional[float] = None) -> ControlOutput:
        p, g = self.p, self.g
        R = quat.to_rotation_matrix(quaternion)
        velocity = np.asarray(velocity, dtype=float)

        # --- disturbance observer (uses the rotors' measured thrust and the attitude, not the wind)
        if g.wind_compensation:
            T_meas = self._last_thrust if measured_thrust_n is None else measured_thrust_n
            a_thrust = R[:, 2] * T_meas / p.mass_kg - np.array([0.0, 0.0, GRAVITY])
            if self._vhat is None:
                self._vhat = velocity.copy()
            err = velocity - self._vhat
            w0 = g.obs_bandwidth_radps
            self._vhat = self._vhat + dt * (a_thrust + self._dhat + 2.0 * w0 * err)
            self._dhat = self._dhat + dt * (w0 * w0 * err)
            n = np.linalg.norm(self._dhat)
            if n > p.max_accel_mps2:
                self._dhat *= p.max_accel_mps2 / n
        kp, kd, ki = (np.asarray(x) for x in (g.kp_pos, g.kd_vel, g.ki_pos))
        if g.wind_compensation:
            ki = ki * g.observer_integral_scale
        e_p = np.asarray(sp.position) - position
        e_v = np.asarray(sp.velocity) - velocity

        # --- position/velocity loop -> commanded acceleration
        # Anti-windup: integrate only near the target and not while the mixer is saturated.
        if not self._last_saturated and np.linalg.norm(e_p) < g.i_active_radius_m:
            self._i_term = np.clip(self._i_term + ki * e_p * dt, -g.i_limit_mps2, g.i_limit_mps2)
        a_cmd = np.asarray(sp.acceleration) + kp * e_p + kd * e_v + self._i_term
        if g.wind_compensation:
            a_cmd = a_cmd - self._dhat          # cancel the estimated drag/wind acceleration
        a_lim = a_cmd.copy()
        n = np.linalg.norm(a_lim)
        limited = bool(n > p.max_accel_mps2)
        if limited:
            a_lim *= p.max_accel_mps2 / n

        # --- desired force vector, tilt-limited, kept upright
        F = p.mass_kg * (a_lim + np.array([0.0, 0.0, GRAVITY]))
        F[2] = max(F[2], 0.2 * p.weight_n)
        fh = np.linalg.norm(F[:2])
        fh_max = F[2] * np.tan(np.radians(p.max_tilt_deg))
        if fh > fh_max:
            F[:2] *= fh_max / fh
            limited = True

        # --- desired attitude (z_b along F, heading from yaw setpoint)
        z_b = F / np.linalg.norm(F)
        x_c = np.array([np.cos(sp.yaw), np.sin(sp.yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        R_des = np.column_stack([x_b, y_b, z_b])

        # collective thrust = force projected on the current body z axis
        thrust = max(float(F @ R[:, 2]), 0.0)

        # --- attitude error (Lee et al.) -> desired body rates
        M = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])
        w_des = np.clip(-np.asarray(g.k_att) * e_R, -g.max_rate_radps, g.max_rate_radps)

        # --- rate loop: angular acceleration command + gyroscopic cancellation
        w = np.asarray(body_rates)
        torque = self._J @ (np.asarray(g.k_rate) * (w_des - w)) + np.cross(w, self._J @ w)

        T_i, T_used, tau_used, sat = desaturate(thrust, torque, p)
        self._last_saturated = sat
        self._last_thrust = T_used
        return ControlOutput(motors.thrust_to_command(T_i, p), T_used, tau_used, a_cmd,
                             _rot_to_quat(R_des), sat, limited)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [w,x,y,z] (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        v = [0.0] * 3
        v[i], v[j], v[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        q = [(R[k, j] - R[j, k]) / s, *v]
    return quat.normalize(np.array(q))
