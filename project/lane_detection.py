import numpy as np
from scipy.ndimage import sobel

class LaneDetection:
    def __init__(self, threshold_rel=1):
        # Class initialization with an optional relative threshold (not used).
        self.threshold_rel = threshold_rel
        # Debug image for visualizing detection results.
        self.debug_image = None

    def detect(self, image: np.ndarray):
        """
        Calculates the left and right lane boundaries based on the given image

        :param image: Image on which the lane is to be detected
        :return: The debug image and the boundaries as arrays
        """
        # Main method for detecting lanes in the given image.
        height, width, _ = image.shape
        # Ignores the upper part of the image to analyze only relevant areas.
        ignore_height = int(height * 0.31)
        image_cropped = image[:height - ignore_height, :]
        # Conversion of the image to grayscale.
        gray = self.rgb_to_gray(image_cropped)
        # Application of the Sobel operator for edge detection.
        edges = self.detect_edges(gray)
        # Conversion of edges into an array of integers for processing.
        edge_array = self.edges_to_array(edges)
        # Extraction of the starting points of the lane markings.
        initial_points = self.extract_lane_edges(edge_array)
        # Analysis and classification of edge points for left and right lanes.
        left_edges, right_edges = self.get_edge_coordinates(edge_array, initial_points)
        # Creation of the debug image with initial marking points.
        self.debug_image = self.create_debug_image(edges, initial_points)
        # Coloring of the debug image based on classified edges.
        self.spread_colors(left_edges, right_edges)
        # Return of the processed debug image.
        return self.debug_image, left_edges, right_edges

    def rgb_to_gray(self, rgb):
        """
        Converts the given image from RGB color space to grayscale

        :param rgb: Color image
        :return: The image in grayscale
        """
        # Conversion of an RGB image into a grayscale image.
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def detect_edges(self, gray: np.ndarray, threshold: int = 63):
        """
        Examines the given image for edges

        :param gray: Image on which the lane is to be detected
        :param threshold: Threshold factor for edge detection
        :return: edges as an array
        """
        # Application of the Sobel operator in x and y directions and combination of the results.
        sobel_x = sobel(gray, axis=1)
        sobel_y = sobel(gray, axis=0)
        # Calculation of total edge strength and binarization based on a threshold.
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edges = (edges > threshold) * 1
        return edges

    def edges_to_array(self, edges):
        """
        Converts the given edges into an array

        :param edges: The edges
        :return: edges as an integer array
        """
        # Conversion of the edge image into an integer array for further processing.
        return edges.astype(np.int32)

    def extract_lane_edges(self, edge_array):
        """
        Detects the original left and right points at the bottom edge of the roadway

        :param edge_array: The edges as an integer array
        :return: the origin points for left and right
        """
        # Determines the lowest points of the left and right lane markings.
        height, width = edge_array.shape
        mid = width // 2
        # Searches for the extreme edge points at the bottom edge of the image.
        bottom_indices = np.where(edge_array[height - 1, :] == 1)[0]
        leftmost_index = max((index for index in bottom_indices if index < mid), default=None)
        rightmost_index = min((index for index in bottom_indices if index > mid), default=None)
        return [(height - 1, leftmost_index, 'left'), (height - 1, rightmost_index, 'right')]

    def get_edge_coordinates(self, edge_array, initial_points, min_distance=2):
        """
        Assigns detected edges to left and right

        :param edge_array: The edges as an integer array
        :param initial_points: The starting points for left and right
        :param min_distance: The maximum distance a point can have for assignment
        :return: an array with left points and an array with right points
        """
        # Generates lists of coordinates for left and right edges.
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        left_edges, right_edges = [], []
        left_y_coords = {}
        right_y_coords = {}

        queue = [(point[1], point[0], point[2]) for point in initial_points if point[1] is not None]
        while queue:
            x, y, side = queue.pop(0)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < edge_array.shape[1] and 0 <= ny < edge_array.shape[0] and edge_array[ny, nx] == 1:
                    edge_array[ny, nx] = 0  # Mark as visited
                    if side == 'left':
                        # Check if a point at the same height is within the minimum distance
                        if not any(nx - min_distance <= existing_x <= nx + min_distance for existing_x in
                                   left_y_coords.get(ny, [])):
                            if ny not in left_y_coords:
                                left_y_coords[ny] = []
                            left_y_coords[ny].append(nx)
                            left_edges.append((nx, ny))
                    else:
                        # Check if a point at the same height is within the minimum distance
                        if not any(nx - min_distance <= existing_x <= nx + min_distance for existing_x in
                                   right_y_coords.get(ny, [])):
                            if ny not in right_y_coords:
                                right_y_coords[ny] = []
                            right_y_coords[ny].append(nx)
                            right_edges.append((nx, ny))
                    queue.append((nx, ny, side))
        return left_edges, right_edges

    def spread_colors(self, left_edges, right_edges):
        """
        Provides a colorful output of the boundaries in the debug image

        :param left_edges: Array with points for left edge
        :param right_edges: Array with points for right edge
        """
        # Colors the lane edges in the debug image.
        for x, y in left_edges:
            self.debug_image[y, x] = [255, 0, 0]  # Red for the left lane
        for x, y in right_edges:
            self.debug_image[y, x] = [0, 0, 255]  # Blue for the right lane

    def create_debug_image(self, edges, initial_points):
        """
        Creates the debug image based on the edges and the origin points

        :param edges: Array where the edges are stored
        :param initial_points: The two origin points for left and right
        :return: debug_image returns the created image
        """
        # Initializes the debug image and marks the starting points.
        debug_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for point in initial_points:
            y, x, side = point
            if x is not None:
                color = [255, 0, 0] if side == 'left' else [0, 0, 255]
                debug_image[y, x] = color
        return debug_image
