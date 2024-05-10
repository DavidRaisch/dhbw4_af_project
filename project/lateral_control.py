from __future__ import annotations

import numpy as np
import math


class LateralControl:

    def __init__(self, controller_type="pure_pursuit", **kwargs): #choose between pure_pursuit and stanley | pure_pursuit better controller
        self.controller_type = controller_type
        self._car_position = np.array([48, 64])

        if controller_type == "stanley":
            # Stanley Regler Parameter
            self.k = 0.25  # control gain
            self.k_soft = 0.7  # softening factor
            self.delta_max = np.pi / 8  # max steering angle
            self.step = 0  # step counter
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

    def stanley_control (self, trajectory, speed): #Works only with info['trajectory'] => Replace 'trajectory' with info['trajectory'] in car.py.
        # Check if the trajectory is empty - error handling
        if len(trajectory) == 0:
            print("Trajectory is empty")
            return 0
        
        # Calculate the cross-track error
        cross_track_error , lookahead_index = self.calculate_cte(trajectory)

        heading_angle = np.arctan2(trajectory[lookahead_index + 1, 1] - trajectory[lookahead_index, 1], trajectory[lookahead_index + 1, 0] - trajectory[lookahead_index, 0])    
        current_heading_angle = np.arctan2(self._car_position[1] - trajectory[0, 1], self._car_position[0] - trajectory[0, 0])
        heading_error = heading_angle - current_heading_angle
        # Calculate the steering angle
        steering_angle = np.arctan2(self.k * cross_track_error, speed + self.k_soft) + heading_error

        # Limit the steering angle
        steering_angle = np.clip(steering_angle, -self.delta_max, self.delta_max)

        self.step += 1 
        return steering_angle
    
    def calculate_cte(self, trajectory):
        # Calculate the distance to the trajectory
        distance = np.linalg.norm(trajectory - self._car_position, axis=1)

        # index of the lookahead point
        lookahead_index = np.argmin(np.abs(distance))

        # Calculate the cross-track error
        cross_track_error = distance[lookahead_index]

        return cross_track_error, lookahead_index
