import numpy as np
from scipy.ndimage import gaussian_filter, sobel


class LaneDetection:
    def __init__(self, ignore_bottom_fraction=0.32, threshold_rel=0.5):
        self.ignore_bottom_fraction = ignore_bottom_fraction
        self.threshold_rel = threshold_rel
        self.debug_image = None

    def detect(self, image: np.ndarray):
        height, width, _ = image.shape
        ignore_height = int(height * self.ignore_bottom_fraction)
        image_cropped = image[:height - ignore_height, :]

        lane_mask = self.isolate_lane(image_cropped)
        edges = self.detect_edges(lane_mask)

        self.debug_image = self.create_debug_image(edges, image, width)
        return edges

    def isolate_lane(self, image: np.ndarray):
        gray = self.rgb_to_gray(image)
        lane_colors = gray[gray < np.percentile(gray, 85)]
        thresh = np.mean(lane_colors) + np.std(lane_colors) * self.threshold_rel
        return gray < thresh

    def detect_edges(self, mask: np.ndarray):
        edges = sobel(mask, axis=0) ** 2 + sobel(mask, axis=1) ** 2
        edges = np.sqrt(edges)
        edges /= edges.max()
        return edges

    def rgb_to_gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def create_debug_image(self, edges, original_image, width):
        debug_image = np.zeros((original_image.shape[0], width, 3), dtype=np.uint8)

        # Calculate the midpoint of the visible (non-ignored) part of the image
        visible_width = width
        center_of_lane = visible_width // 2
        # Apply colors to the left and right edges
        edge_left = edges[:, :center_of_lane] > 0
        edge_right = edges[:, center_of_lane:] > 0

        # Color the left and right edges
        debug_image[:edges.shape[0], :center_of_lane][edge_left] = [255, 0, 0]
        debug_image[:edges.shape[0], center_of_lane:][edge_right] = [0, 0, 255]

        # Pad the bottom part of the image to restore the ignored sections
        padding = np.zeros((0, width, 3), dtype=np.uint8)
        complete_debug_image = np.vstack((debug_image, padding))
        return complete_debug_image

# This code replaces the LaneDetection class in your lane_detection.py file.
