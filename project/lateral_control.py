from __future__ import annotations

import numpy as np


class LateralControl:

    def __init__(self, controller_type="stanley", **kwargs):
        self.controller_type = controller_type
        self._car_position = np.array([48, 64])

        if controller_type == "stanley":
            # Stanley Regler Parameter
            self.k_p = kwargs.get('k_p', 0.1)
            self.k_i = kwargs.get('k_i', 0.01)
            self.k_d = kwargs.get('k_d', 0.01)
            self.integral_cte = 0.0  # Initialisierung des integralen Anteils des Reglers


    def control(self, trajectory, speed):
        if self.controller_type == "stanley":
            return self.stanley_control(trajectory, speed)
        else:
            raise ValueError("Unsupported controller type")

    def stanley_control(self, trajectory, speed):
        if not trajectory.size:
            return 0.0  # Wenn die Trajektorie leer ist, gibt es keinen Lenkwinkel

        # Vehicle parameters
        L = 0.5  # Distance between front and rear axle (vehicle length)
        k = 0.00000000000001  # Gain factor for steering angle calculation
        k_p = 0.105  # Proportional gain for heading error
        max_delta = np.pi / 6  # Maximum steering angle

        # Current vehicle position
        car_x, car_y = self._car_position

        # Find the closest point on the trajectory to the vehicle
        min_dist = float('inf')
        closest_point = None
        for point in trajectory:
            dist = np.linalg.norm(np.array(point) - self._car_position)
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        # Calculate heading error (angle difference between trajectory and vehicle heading)
        trajectory_yaw = np.arctan2(trajectory[-1][1] - trajectory[0][1], trajectory[-1][0] - trajectory[0][0])
        heading_error = trajectory_yaw - np.arctan2(car_y - closest_point[1], car_x - closest_point[0])
        heading_error = self.get_valid_angle(heading_error)

        # Calculate cross-track error (distance between vehicle and trajectory)
        e_r = min_dist

        # Calculate delta error (additional steering angle correction based on cross-track error)
        delta_error = np.arctan(k * e_r / (L * speed + 1.0e-6))

        # Calculate total steering angle with proportional gain for heading error
        steering_angle = k_p * heading_error + delta_error

        # Limit the steering angle to avoid extreme values
        steering_angle = np.clip(steering_angle, -max_delta, max_delta)

        return steering_angle



    
    def get_valid_angle(self, angle):
        """
        Adjusts the angle to be within the range of -pi to pi.
        """
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle





