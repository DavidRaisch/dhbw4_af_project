import numpy as np

class PathPlanning:
    def __init__(self):
        # Constructor of the PathPlanning class without specific attributes.
        pass

    def plan(self, left_lane_points, right_lane_points):
        """
        Calculates the centerline and curvature based on the left and right lane boundaries.

        :param left_lane_points: Coordinates of the left lane boundary
        :param right_lane_points: Coordinates of the right lane boundary
        :return: List of centerline points, reduced to every 10th point, and the average curvature
        """
        # Conversion of input lists into NumPy arrays
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)

        # Calculation of midpoints between left and right points
        midpoints = self.calculate_midpoints(left_lane_points, right_lane_points)

        # Calculation of curvature based on these midpoints
        curvature = self.calculate_curvature(midpoints)

        # Filtering out erroneous calculations
        if curvature > 25:
            curvature = 25

        # Selection of every 10th point of the centerline for output
        sampled_midpoints = midpoints[::10]

        return sampled_midpoints, curvature

    def calculate_midpoints(self, left_lane_points, right_lane_points):
        """
        Calculates the midpoints of the lane.

        :param left_lane_points: Array of coordinates of the left lane boundary
        :param right_lane_points: Array of coordinates of the right lane boundary
        :return: Array of calculated midpoints
        """
        if not left_lane_points.any() or not right_lane_points.any():
            # Returns an empty array if any of the input arrays are empty
            return np.array([])

        # Determining the shorter length of the two lists of points
        min_len = min(len(left_lane_points), len(right_lane_points))

        # Extraction of x and y coordinates
        left_x, left_y = left_lane_points[:min_len].T
        right_x, right_y = right_lane_points[:min_len].T

        # Calculation of midpoints
        mid_x = (left_x + right_x) / 2
        mid_y = (left_y + right_y) / 2
        midpoints = np.column_stack((mid_x, mid_y))

        return midpoints

    def calculate_curvature(self, points):
        """
        Calculates the curvature of the points on the centerline after a pre-analysis that considers only the continuous line,
        starting from the bottommost position (highest y-value) moving upwards without interruptions of more than one pixel.

        :param points: Array of midpoints
        :return: Average curvature
        """
        try:
            if points.size == 0:
                # If there are no points, the curvature is 0
                return 0

            # Sorting the points by y-coordinate from bottom to top (descending order)
            points = points[np.argsort(-points[:, 1])]

            # Filtering the points to ensure only the bottommost line is used
            filtered_points = [points[0]]
            for i in range(1, len(points)):
                # Add only points that are at most 1 pixel vertically and 5 pixels horizontally away from the previous point
                if abs(points[i][1] - filtered_points[-1][1]) <= 1 and abs(points[i][0] - filtered_points[-1][0]) <= 5:
                    filtered_points.append(points[i])
                elif points[i][1] < filtered_points[-1][1] - 1:
                    break  # Break in the line
            filtered_points = np.array(filtered_points)

            if filtered_points.size == 0:
                return 0

            # Calculation of first and second derivatives
            dx_dt = np.gradient(filtered_points[:, 0])
            dy_dt = np.gradient(filtered_points[:, 1])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            # Ensuring the denominator does not become too small
            denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
            safe_denominator = np.where(denominator < 1e-6, 1e-6, denominator)

            # Calculation of curvature
            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / safe_denominator
            curvature = curvature * 10  # Scaling of curvature for better readability

            return curvature.mean()
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0
