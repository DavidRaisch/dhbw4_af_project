from __future__ import annotations

import numpy as np
import time

class LongitudinalControl:
    def __init__(self):
        # define init values
        self.kp = 0.1
        self.ki = 0.0001
        self.kd = 0.001
        self.error_integral = 0
        self.prev_error = 0
        self.start_time = time.time()

    def control(self, current_speed: float, target_speed: float, steering_angle: float) -> tuple[float, float]:
    # Here, the control of acceleration and braking is carried out.
        error = target_speed - current_speed
        self.error_integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        control_signal = self.kp * error  + self.kd * derivative + self.ki * self.error_integral

        # Calculation of acceleration and braking based on the control signal.
        if control_signal > 0:
            acceleration = min(control_signal, 1.0)  # Limiting the maximum value to 1.0.
            brake = 0.0
        else:
            acceleration = 0.0
            brake = min(-control_signal, 1.0)  # Limiting the maximum value to 1.0.

        

        return acceleration, brake

    def predict_target_speed(self, curvature, steeringangle):
        # Here, the target speed is determined based on the curvature of the curve.
       
        timer = time.time() - self.start_time # Wait for zooming in at the beginning.
        if timer < 1:
            target_speed = 20.0
        else:
            #calculate speed depending on curvature
            if curvature > 50: 
                target_speed = 50.0 - curvature*0.3
            elif curvature > 20:
                target_speed = 50.0 - curvature*0.4
            elif curvature < 20:
                target_speed = 50 - curvature*0.5
            else:
                target_speed = 25

        return target_speed

""" #Not needed => included for testing purposes only.
    def _calculate_curvature(self, trajectory: list[tuple[float, float]]) -> float:
        # Berechnung der Kurvenkrümmung basierend auf den Trajektoriepunkten
        x = np.array([p[0] for p in trajectory])
        y = np.array([p[1] for p in trajectory])
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5)

        curvature = abs(curvature) * 1000
        return np.mean(curvature)
"""
