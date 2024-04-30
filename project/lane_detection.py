import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import find_peaks

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
        vertical_edges = sobel(mask, axis=0) ** 2
        horizontal_edges = sobel(mask, axis=1) ** 2
        edges = np.sqrt(vertical_edges + horizontal_edges)
        edges /= edges.max()
        return edges

    def rgb_to_gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def create_debug_image(self, edges, width):
        debug_image = np.zeros((edges.shape[0], width, 3), dtype=np.uint8)
        prev_peaks = None

        # Find and color the actual edge paths
        for row in reversed(range(edges.shape[0])):
            row_peaks, _ = find_peaks(edges[row], height=0.3)
            corrected_peaks = self.correct_peak_assignment(row_peaks, prev_peaks)

            if corrected_peaks.size >= 2:
                # Assign edges based on proximity to previous peaks
                left_index = corrected_peaks[0]  # first peak from the left
                right_index = corrected_peaks[-1]  # last peak from the right

                # Mark the detected edges in the image
                debug_image[row, left_index] = [255, 0, 0]  # Red for left edge
                debug_image[row, right_index] = [0, 0, 255]  # Blue for right edge

                prev_peaks = corrected_peaks
            else:
                # Handle cases where peaks may be missed due to sharp curves
                if prev_peaks is not None and corrected_peaks.size == 1:
                    # Attempt to determine if the single peak is left or right based on previous data
                    if corrected_peaks[0] - prev_peaks[0] < prev_peaks[-1] - corrected_peaks[0]:
                        debug_image[row, corrected_peaks[0]] = [255, 0, 0]  # Assume left if closer to previous left
                        prev_peaks = np.array([corrected_peaks[0], prev_peaks[-1]])
                    else:
                        debug_image[row, corrected_peaks[0]] = [0, 0, 255]  # Assume right if closer to previous right
                        prev_peaks = np.array([prev_peaks[0], corrected_peaks[0]])
                else:
                    prev_peaks = None

        return debug_image

    def correct_peak_assignment(self, peaks, prev_peaks):
        if prev_peaks is None or len(prev_peaks) < 2:
            return np.array(peaks)

        # Determine left and right peaks from previous row
        left_prev_peak = prev_peaks[0]
        right_prev_peak = prev_peaks[-1]

        corrected_peaks = []
        for peak in peaks:
            if abs(peak - left_prev_peak) < abs(peak - right_prev_peak):
                corrected_peaks.append(peak)
            else:
                corrected_peaks.append(peak)

        corrected_peaks = np.array(corrected_peaks)
        if len(corrected_peaks) >= 2:
            # Avoid single peak being misclassified
            if np.all(corrected_peaks <= left_prev_peak + 10):
                corrected_peaks = np.append(corrected_peaks, right_prev_peak)
            elif np.all(corrected_peaks >= right_prev_peak - 10):
                corrected_peaks = np.insert(corrected_peaks, 0, left_prev_peak)

        return corrected_peaks
