import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        self.canny_low_threshold = 150
        self.canny_high_threshold = 450
        self.ignore_bottom_fraction = 0.32

    def detect(self, image: np.ndarray) -> np.ndarray:
        ignore_height = int(image.shape[0] * self.ignore_bottom_fraction)
        image[-ignore_height:, :] = 0

        edges = cv2.Canny(image, self.canny_low_threshold, self.canny_high_threshold)
        self.debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Schritt 3: Finden der linken und rechten Kantenpunkte
        left_lane_edges = edges[:, :edges.shape[1] // 2]
        right_lane_edges = edges[:, edges.shape[1] // 2:]

        # Schritt 4: Zuordnen der erkannten Kanten zur linken und rechten Fahrspurbegrenzung
        left_lane_points = np.argwhere(left_lane_edges == 255)
        right_lane_points = np.argwhere(right_lane_edges == 255)

        # Farbe der linken Kantenpunkte rot machen
        for point in left_lane_points:
            self.debug_image[point[0], point[1], :] = [0, 0, 255]  # BGR-Farbformat: Rot

        # Farbe der rechten Kantenpunkte blau machen
        for point in right_lane_points:
            self.debug_image[point[0], point[1] + edges.shape[1] // 2, :] = [255, 0, 0]  # BGR-Farbformat: Blau

        return edges, self.debug_image

"""
    def detect_lines(self, image: np.ndarray) -> list:
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
        print("Detected lines:", lines)
        return lines if lines is not None else []

    def classify_lines(self, lines: list, image_width: int) -> tuple:
        left_lines = []
        right_lines = []
        print("Lines detected:", lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Berechne die Steigung der Linie
            slope = (y2 - y1) / (x2 - x1)
            print("Slope:", slope)
            # Basierend auf der Steigung klassifiziere die Linie als links oder rechts
            if slope < 0 and x1 < image_width / 2 and x2 < image_width / 2:
                left_lines.append(line)
            elif slope > 0 and x1 > image_width / 2 and x2 > image_width / 2:
                right_lines.append(line)
        return left_lines, right_lines

    def detect(self, image: np.ndarray) -> tuple:
        #print("Detect method called")
        preprocessed = self.preprocess_image(image)
        lines = self.detect_lines(preprocessed)
        #print("Lines detected:", lines)
        if lines:
            image_width = image.shape[1]
            left_lines, right_lines = self.classify_lines(lines, image_width)
            return left_lines, right_lines
        else:
            return [], []

"""
    
