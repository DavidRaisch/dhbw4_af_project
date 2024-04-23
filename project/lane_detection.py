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

        self.debug_image = self.create_debug_image(edges, width)
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

    def create_debug_image(self, edges, width):
        debug_image = np.zeros((edges.shape[0], width, 3), dtype=np.uint8)

        # Find and color the actual edge paths
        for row in range(edges.shape[0]):
            if np.any(edges[row] > 0):  # Check if there's any edge in the row
                left_index = np.argmax(edges[row] > 0)
                right_index = width - np.argmax(edges[row][::-1] > 0) - 1

                # Mark the detected edges in the image
                debug_image[row, left_index] = [255, 0, 0]  # Red for left edge
                debug_image[row, right_index] = [0, 0, 255]  # Blue for right edge

        return debug_image
