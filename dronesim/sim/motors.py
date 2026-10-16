"""Motor/rotor model and the thrust<->wrench mixer.

Four distinct quantities are kept apart:
    command u in [0,1]   -> what the controller outputs (what the graphs show)
    angular speed w      -> rad/s, first-order lag behind the command target
    thrust T = kT w^2    -> N
    electrical power     -> W, P = Q*w/eff + idle,  Q = kM w^2
"""
import numpy as np

from dronesim.sim.params import DroneParams


def clip_command(u) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), 0.0, 1.0)


def command_to_omega(u, p: DroneParams) -> np.ndarray:
    """Target angular speed for a command (linear map; thrust is therefore quadratic in u)."""
    return clip_command(u) * p.omega_max_radps


def thrust(omega, p: DroneParams) -> np.ndarray:
    return p.k_thrust * np.asarray(omega, dtype=float) ** 2


def drag_torque(omega, p: DroneParams) -> np.ndarray:
    return p.k_torque * np.asarray(omega, dtype=float) ** 2


def electrical_power(omega, p: DroneParams) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    return drag_torque(omega, p) * omega / p.motor_efficiency + p.motor_idle_power_w


def thrust_to_command(T, p: DroneParams) -> np.ndarray:
    """Inverse static map: per-motor thrust (N) -> command, clipped to [0,1]."""
    T = np.maximum(np.asarray(T, dtype=float), 0.0)
    return clip_command(np.sqrt(T / p.max_thrust_per_motor_n))


def mixer_matrix(p: DroneParams) -> np.ndarray:
    """A such that [F_total, tau_x, tau_y, tau_z] = A @ [T1..T4]  (steady-state, ignores rotor spin-up).

    tau_x = sum(y_i T_i), tau_y = -sum(x_i T_i), tau_z = -sum(spin_i * kM/kT * T_i)
    """
    xy = p.motor_positions
    ratio = p.yaw_torque_ratio_m
    return np.array([
        np.ones(4),
        xy[:, 1],
        -xy[:, 0],
        -p.spin_dir * ratio,
    ])


def thrusts_from_wrench(wrench, p: DroneParams) -> np.ndarray:
    """Per-motor thrusts (N) realising [F_total, tau_x, tau_y, tau_z]; may be negative/over-limit
    (the caller handles saturation)."""
    return np.linalg.solve(mixer_matrix(p), np.asarray(wrench, dtype=float))
