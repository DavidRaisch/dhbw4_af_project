import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        # Schwellenwerte für die Erkennung signifikanter Farbunterschiede
        self.canny_low_threshold = 150
        self.canny_high_threshold = 450
        self.ignore_bottom_fraction = 0.32  # 40% des Bildes unten ignorieren

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Ignorieren des schwarzen Bereichs am unteren Rand
        ignore_height = int(image.shape[0] * self.ignore_bottom_fraction)
        image[-ignore_height:, :] = 0

        # Verarbeitung des Bildes zur Extraktion von signifikanten Kanten
        self.debug_image = np.zeros_like(image)
        edges = cv2.Canny(image, self.canny_low_threshold, self.canny_high_threshold)
        self.debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges

    def detect_lines(self, image: np.ndarray) -> list:
        # Linienfindung im bearbeiteten Bild
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
        return lines if lines is not None else []

    def detect(self, image: np.ndarray) -> list:
        # Anwendung der Verarbeitungspipeline
        preprocessed = self.preprocess_image(image)
        lines = self.detect_lines(preprocessed)
        return lines

    def visualize_debug_image(self):
        # Visualisierung des verarbeiteten Bildes
        if self.debug_image is not None:
            cv2.imshow('Debug Image - Significant Edges', self.debug_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
