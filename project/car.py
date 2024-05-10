from __future__ import annotations

import cv2
import numpy as np

from lane_detection import LaneDetection
from lateral_control import LateralControl
from longitudinal_control import LongitudinalControl
from path_planning import PathPlanning


class Car:

    def __init__(self):
        self._lane_detection = LaneDetection()
        self._path_planning = PathPlanning()
        self._lateral_control = LateralControl()
        self._longitudinal_control = LongitudinalControl()

    def next_action(self, observation: np.ndarray, info: dict[str, any]) -> list:
        """Defines the next action to take based on the current observation, reward, and other information.

        Args:
            observation (np.ndarray): The current observation of the environment.
            info (dict[str, Any]): Additional information about the environment.

        Returns:
            List: The action to take:
                0: steering, -1 is full left, +1 is full right
                1: gas, 0 is no gas, 1 is full gas
                2: breaking, 0 is no break, 1 is full break
        """
        debug_image, left_lane_boundaries, right_lane_boundaries = self._lane_detection.detect(observation)
        trajectory, curvature = self._path_planning.plan(left_lane_boundaries, right_lane_boundaries)
        cv_image = np.asarray(debug_image, dtype=np.uint8)
        sampled_midpoints = np.array(trajectory, dtype=np.int32)
        for point in sampled_midpoints:
            if 0 < point[0] < cv_image.shape[1] and 0 < point[1] < cv_image.shape[0]:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 255]

        cv2.putText(cv_image, f"Krümmung: {curvature:.2f}", (15, 63), cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)
        cv_image = np.asarray(debug_image, dtype=np.uint8)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Car Racing - Lane Detection', cv_image)
        cv2.waitKey(1)

        steering_angle = self._lateral_control.control(trajectory, info['speed'])
        target_speed = self._longitudinal_control.predict_target_speed(curvature, steering_angle) #wieder in curvature ändern anstatt info ['trajectory']
        acceleration, braking = self._longitudinal_control.control(info['speed'], target_speed, steering_angle)

        action = [steering_angle, acceleration, braking]

        return action
