import numpy as np

class PathPlanning:
    def __init__(self):
        pass

    def plan(self, left_lane_points, right_lane_points):
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)
        midpoints = self.calculate_midpoints(left_lane_points, right_lane_points)
        curvature = self.calculate_curvature(midpoints)
        return midpoints.tolist(), curvature

    def calculate_midpoints(self, left_lane_points, right_lane_points):
        min_len = min(len(left_lane_points), len(right_lane_points))
        left_x, left_y = left_lane_points[:min_len].T
        right_x, right_y = right_lane_points[:min_len].T
        mid_x = (left_x + right_x) / 2
        mid_y = (left_y + right_y) / 2
        midpoints = np.column_stack((mid_x, mid_y))
        return midpoints

    def calculate_curvature(self, points):
        # Assuming points are evenly spaced
        dx_dt = np.gradient(points[:, 0])
        dy_dt = np.gradient(points[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
        return curvature.mean()


