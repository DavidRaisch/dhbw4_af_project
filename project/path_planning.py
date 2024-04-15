import numpy as np

class PathPlanning:
    def __init__(self):
        pass

    def plan(self, left_lane_points, right_lane_points):
        # Convert inputs to numpy arrays
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)

        # Check if either array is empty
        if left_lane_points.size == 0 or right_lane_points.size == 0:
            return [], 0  # Return empty path and zero curvature if no points

        # Interpolation to match points on a common y-coordinate set
        y_min = max(np.min(left_lane_points[:, 1]), np.min(right_lane_points[:, 1]))
        y_max = min(np.max(left_lane_points[:, 1]), np.max(right_lane_points[:, 1]))
        common_y = np.linspace(y_min, y_max, num=max(left_lane_points.shape[0], right_lane_points.shape[0]))

        # Create interpolation functions for each lane
        left_interp = interp1d(left_lane_points[:, 1], left_lane_points[:, 0], kind='linear', fill_value='extrapolate')
        right_interp = interp1d(right_lane_points[:, 1], right_lane_points[:, 0], kind='linear', fill_value='extrapolate')

        # Interpolate x coordinates on the common y values
        left_x = left_interp(common_y)
        right_x = right_interp(common_y)

        # Calculate midpoints
        midpoints = np.column_stack(((left_x + right_x) / 2, common_y))

        # Calculate curvature
        curvature = self.calculate_curvature(midpoints)

        return midpoints.tolist(), curvature

    def calculate_curvature(self, points):
        if len(points) < 2:
            return 0  # Not enough points to define a line

        start_point = points[0]
        end_point = points[-1]

        # Calculate the differences in the coordinates
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Calculate the angle in radians between the line and the y-axis
        angle_radians = np.arctan2(dx, dy)  # arctan2 handles the quadrant correctly

        # Convert angle from radians to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

# Ensure that the scipy.interpolate.interp1d import statement is included at the beginning of your file
from scipy.interpolate import interp1d
