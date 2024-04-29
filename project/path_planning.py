import numpy as np
from scipy.interpolate import interp1d

class PathPlanning:
    def __init__(self):
        pass

    def plan(self, left_lane_points, right_lane_points):
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)

        if left_lane_points.size == 0 or right_lane_points.size == 0:
            return [], 0

        y_min = max(np.min(left_lane_points[:, 1]), np.min(right_lane_points[:, 1]))
        y_max = min(np.max(left_lane_points[:, 1]), np.max(right_lane_points[:, 1]))
        common_y = np.linspace(y_min, y_max, num=max(left_lane_points.shape[0], right_lane_points.shape[0]))

        left_interp = interp1d(left_lane_points[:, 1], left_lane_points[:, 0], kind='linear', fill_value='extrapolate')
        right_interp = interp1d(right_lane_points[:, 1], right_lane_points[:, 0], kind='linear', fill_value='extrapolate')

        left_x = left_interp(common_y)
        right_x = right_interp(common_y)

        midpoints = np.column_stack(((left_x + right_x) / 2, common_y))

        curvature = self.calculate_curvature(midpoints)

        return midpoints.tolist(), curvature

    def calculate_curvature(self, points):
        if len(points) < 2:
            return 0  # Not enough points to define a direction

        start_point = points[0]
        end_point = points[-1]

        # Vektor vom Start- zum Endpunkt
        vector_x = end_point[0] - start_point[0]
        vector_y = end_point[1] - start_point[1]

        # Winkel zur Y-Achse
        angle = np.arctan2(vector_x, vector_y)  # arctan2(x, y) gibt den Winkel im Bogenmaß zurück
        angle_degrees = np.degrees(angle)  # Umwandlung in Grad

        # Runden des Winkels auf die nächste ganze Zahl
        try:
            rounded_angle = round(angle_degrees)
        except:
            rounded_angle = 0

        return rounded_angle


