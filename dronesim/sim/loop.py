"""Closed-loop simulation driver: controller at a fixed rate, physics at a fixed (faster) step.

Commands are held (zero-order hold) between controller updates. History is returned as plain
arrays so it can be recorded, replayed and plotted without re-running anything.
"""
from typing import Callable

import numpy as np

from dronesim.sim import motors
from dronesim.sim.controller import FlightController, Setpoint
from dronesim.sim.quadrotor import Quadrotor


def simulate(quad: Quadrotor, ctrl: FlightController, setpoint_fn: Callable[[float], Setpoint],
             duration_s: float, control_hz: float = 250.0, record_every: int = 1) -> dict:
    """Run for `duration_s`. `setpoint_fn(t)` gives the setpoint at time t. Returns a history dict."""
    n_steps = int(round(duration_s / quad.dt))
    per_ctrl = max(1, int(round(1.0 / (control_hz * quad.dt))))
    ctrl_dt = per_ctrl * quad.dt
    hist = {k: [] for k in ("t", "position", "velocity", "quaternion", "body_rates", "command", "omega",
                            "thrust", "power_w", "setpoint_position", "saturated", "limited", "wind", "disturbance_est")}
    u = np.full(4, quad.p.hover_command)
    saturated = limited = False
    for i in range(n_steps):
        sp = setpoint_fn(quad.t)
        if i % per_ctrl == 0:
            measured_thrust = float(motors.thrust(quad.rotor_speeds, quad.p).sum())   # ESC-style rotor telemetry
            out = ctrl.update(quad.position, quad.velocity, quad.quaternion, quad.body_rates, sp, ctrl_dt,
                              measured_thrust)
            u, saturated, limited = out.command, out.saturated, out.limited
        if i % record_every == 0:
            hist["t"].append(quad.t)
            hist["position"].append(quad.position)
            hist["velocity"].append(quad.velocity)
            hist["quaternion"].append(quad.quaternion)
            hist["body_rates"].append(quad.body_rates)
            hist["setpoint_position"].append(np.asarray(sp.position, dtype=float))
            hist["saturated"].append(saturated)
            hist["limited"].append(limited)
            hist["disturbance_est"].append(ctrl.disturbance_estimate)
        o = quad.step(u)
        if i % record_every == 0:
            hist["wind"].append(o.wind_world)
            hist["command"].append(o.command)
            hist["omega"].append(o.omega)
            hist["thrust"].append(o.thrust)
            hist["power_w"].append(o.power_w)
    return {k: np.array(v) for k, v in hist.items()}
