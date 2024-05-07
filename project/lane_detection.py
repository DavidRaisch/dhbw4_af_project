import numpy as np
from scipy.ndimage import sobel

class LaneDetection:
    def __init__(self, threshold_rel=0.5):
        # Initialisierung der LaneDetection-Klasse mit einem Schwellenwert für die Kanten.
        self.threshold_rel = threshold_rel
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Hauptmethode zur Detektion von Fahrspurmarkierungen in einem Bild.
        height, width, _ = image.shape
        # Ignoriere den oberen Teil des Bildes, da Fahrbahnmarkierungen im unteren Teil zu finden sind.
        ignore_height = int(height * 0.33)
        image_cropped = image[:height - ignore_height, :]

        # Konvertiere das zugeschnittene Bild in Graustufen.
        gray = self.rgb_to_gray(image_cropped)
        # Erkenne Kanten im Bild mit einem Sobel-Filter.
        edges = self.detect_edges(gray)
        # Extrahiere die initialen Punkte der linken und rechten Fahrbahnmarkierungen.
        initial_points = self.extract_lane_edges(edges)
        # Erstelle ein Debug-Bild für die Visualisierung.
        self.debug_image = self.create_debug_image(edges, initial_points)
        # Verbreite die Farben der initialen Punkte auf benachbarte Kantenpunkte.
        self.spread_colors(edges, initial_points)
        return self.debug_image

    def rgb_to_gray(self, rgb):
        # Wandelt ein RGB-Bild in ein Graustufenbild um.
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def detect_edges(self, gray: np.ndarray, threshold: int = 120):
        # Nutze den Sobel-Operator zur Kantenfindung in x- und y-Richtung und kombiniere die Ergebnisse.
        sobel_x = sobel(gray, axis=1)
        sobel_y = sobel(gray, axis=0)
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Binarisiere die Kanten anhand eines Schwellenwertes.
        edges = (edges > threshold) * 1
        return edges

    def extract_lane_edges(self, edges):
        # Ermittelt die untersten Punkte der linken und rechten Fahrbahnmarkierungen.
        height, width = edges.shape
        mid = width // 2  # Mittelpunkt der Breite des Bildes
        bottom_indices = np.where(edges[height - 1, :] == 1)[0]
        if len(bottom_indices) > 0:
            # Linke Markierung: Punkt am weitesten rechts, aber links von der Mitte
            leftmost_index = max((index for index in bottom_indices if index < mid), default=None)
            # Rechte Markierung: Punkt am weitesten links, aber rechts von der Mitte
            rightmost_index = min((index for index in bottom_indices if index > mid), default=None)
        else:
            leftmost_index = rightmost_index = None
        return [(height - 1, leftmost_index, 'left'), (height - 1, rightmost_index, 'right')]

    def create_debug_image(self, edges, initial_points):
        # Erstellt ein Debug-Bild, das die initialen Punkte einfärbt.
        debug_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for point in initial_points:
            y, x, side = point
            if x is not None:
                color = [255, 0, 0] if side == 'left' else [0, 0, 255]
                debug_image[y, x] = color
        return debug_image

    def spread_colors(self, edges, initial_points):
        # Verwendet eine einfache Liste als Warteschlange, um die Farben entlang der Kanten zu verbreiten.
        queue = [(point[1], point[0], point[2]) for point in initial_points if point[1] is not None]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y, side = queue.pop(0)
            current_color = [255, 0, 0] if side == 'left' else [0, 0, 255]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < edges.shape[1] and 0 <= ny < edges.shape[0]:
                    if edges[ny, nx] == 1 and np.all(self.debug_image[ny, nx] == 0):
                        self.debug_image[ny, nx] = current_color
                        queue.append((nx, ny, side))
