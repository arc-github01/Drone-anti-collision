"""Virtual quadcopter parameters (SI units).

IMPORTANT: these are ASSUMED, Tello-class order-of-magnitude values, not measured
specifications. Change them freely; nothing here claims to be a real DJI Tello datasheet.
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

GRAVITY = 9.80665  # m/s^2


@dataclass
class DroneParams:
    mass_kg: float = 0.080
    # Motor offsets from the centre of mass along body x (forward) and y (left), metres.
    arm_xy_m: float = 0.045
    # Principal moments of inertia (diagonal), kg m^2.
    inertia_kgm2: Tuple[float, float, float] = (1.0e-4, 1.0e-4, 2.0e-4)

    # Rotor model: T = kT * w^2 (N), reaction/drag torque Q = kM * w^2 (N m).
    omega_max_radps: float = 1500.0
    max_thrust_per_motor_n: float = 0.40          # sets kT: kT = Tmax / omega_max^2
    yaw_torque_ratio_m: float = 0.016             # kM / kT
    rotor_inertia_kgm2: float = 2.0e-7            # spin-axis inertia; reaction torque on body when spinning up
    motor_time_constant_s: float = 0.03           # first-order lag: dw/dt = (w_cmd - w) / tau

    # Electrical power: P = Q*w/eff + idle (per motor). Distinct from thrust and command.
    motor_efficiency: float = 0.7
    motor_idle_power_w: float = 0.3

    # Aerodynamic drag: F = -c_d * |v_rel| * v_rel,  c_d = 0.5 * rho * Cd * A  (kg/m)
    drag_coeff_kg_per_m: float = 0.006
    angular_damping_nms: float = 2.0e-6

    # Control limits used by the controller/planner (constraints, not physics).
    max_tilt_deg: float = 35.0
    max_accel_mps2: float = 6.0

    @property
    def k_thrust(self) -> float:
        return self.max_thrust_per_motor_n / self.omega_max_radps**2

    @property
    def k_torque(self) -> float:
        return self.k_thrust * self.yaw_torque_ratio_m

    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag(self.inertia_kgm2)

    @property
    def weight_n(self) -> float:
        return self.mass_kg * GRAVITY

    @property
    def hover_command(self) -> float:
        """Normalised command [0,1] at which 4 motors exactly support the weight."""
        thrust_each = self.weight_n / 4.0
        return float(np.sqrt(thrust_each / self.max_thrust_per_motor_n))

    # Motor layout (X configuration, body FLU): positions and spin directions.
    # M1 front-right (CCW), M2 rear-right (CW), M3 rear-left (CCW), M4 front-left (CW).
    @property
    def motor_positions(self) -> np.ndarray:
        d = self.arm_xy_m
        return np.array([[d, -d], [-d, -d], [-d, d], [d, d]])

    @property
    def spin_dir(self) -> np.ndarray:
        """+1 = CCW seen from above (+z), -1 = CW. Body reaction torque is opposite to spin."""
        return np.array([1.0, -1.0, 1.0, -1.0])
