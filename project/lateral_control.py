from __future__ import annotations

import numpy as np
import math


class LateralControl:

    def __init__(self, controller_type="pure_pursuit", **kwargs): #choose between pure_pursuit and stanley | pure_pursuit better controller
        self.controller_type = controller_type
        self._car_position = np.array([48, 64])

        if controller_type == "stanley":
            # Stanley Regler Parameter
            self.k_p = kwargs.get('k_p', 0.1)
            self.k_i = kwargs.get('k_i', 0.01)
            self.k_d = kwargs.get('k_d', 0.8)
            self.integral_cte = 0.0  # Initialisierung des integralen Anteils des Reglers
        elif controller_type == "pure_pursuit":
            # Pure Pursuit Regler Parameter
            self.lookahead_distance = kwargs.get('lookahead_distance', 3.0)


    def control(self, trajectory, speed):
        if self.controller_type == "stanley":
            return self.stanley_control(trajectory, speed)
        elif self.controller_type == "pure_pursuit":
            return self.pure_pursuit_control(trajectory, speed)
        else:
            raise ValueError("Unsupported controller type")
        

        
    def pure_pursuit_control(self, trajectory, speed):
        k_p = 0.2
        
        closest_point_index = self._find_closest_point(trajectory, self._car_position)
        lookahead_point_index = self._find_lookahead_point(trajectory, closest_point_index, self._car_position)

        if lookahead_point_index == -1:
            return 0.0  # If no lookahead point found, maintain current angle
        
        lookahead_point = trajectory[lookahead_point_index]

        # Calculate delta between lookahead point and current vehicle position
        y_delta = lookahead_point[1] - self._car_position[1]
        x_delta = lookahead_point[0] - self._car_position[0]

        # Calculate angle between lookahead point and vehicle orientation
        alpha = math.atan2(y_delta, x_delta) - self._car_position[1]

        # Adjust alpha to be within the range of -pi to pi
        if alpha > np.pi / 2:
            alpha -= np.pi
        if alpha < -np.pi / 2:
            alpha += np.pi 

        # Calculate steering output using the given formula
        steer_output = math.atan(2 * 3 * math.sin(alpha) / (k_p * speed))

        # Obey the max steering angle bounds
        max_steering_angle = 1.22
        steer_output = np.clip(steer_output, -max_steering_angle, max_steering_angle)

        return steer_output


    
    def _find_closest_point(self, trajectory, current_position):
        min_distance = float('inf')
        closest_point_index = 0

        for i, point in enumerate(trajectory):
            distance = math.sqrt((point[0] - current_position[0]) ** 2 + (point[1] - current_position[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i
        
        return closest_point_index
    
    def _find_lookahead_point(self, trajectory, closest_point_index, current_position):
        for i in range(closest_point_index + 1, len(trajectory)):
            distance = math.sqrt((trajectory[i][0] - current_position[0]) ** 2 + (trajectory[i][1] - current_position[1]) ** 2)
            if distance > self.lookahead_distance:
                return i
        return -1
    

    
    def stanley_control(self, trajectory, speed):
        if not trajectory.size:
            return 0.0  # Wenn die Trajektorie leer ist, gibt es keinen Lenkwinkel

        # Vehicle parameters
        L = 0.5  # Distance between front and rear axle (vehicle length)
        k = 0.0000000001  # Gain factor for steering angle calculation
        k_p = 0.105  # Proportional gain for heading error #0.105
        max_delta = np.pi / 8  # Maximum steering angle

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
        print(heading_error)


        # Calculate cross-track error (distance between vehicle and trajectory)
        e_r = min_dist

        # Calculate total steering angle with proportional gain for heading error
        steering_angle = k_p * heading_error 

        # Limit the steering angle to avoid extreme values
        steering_angle = np.clip(steering_angle, -max_delta, max_delta)

        return steering_angle

    def get_valid_angle(self, angle):
    
        # Adjusts the angle to be within the range of -pi to pi.
        
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

        



