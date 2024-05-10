import numpy as np
from scipy.ndimage import sobel

class LaneDetection:
    def __init__(self, threshold_rel=1):
        # Initialisierung der Klasse mit einem optionalen relativen Schwellenwert (nicht verwendet).
        self.threshold_rel = threshold_rel
        # Debug-Bild für die Visualisierung der Erkennungsergebnisse.
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Hauptmethode zur Erkennung von Fahrspuren im gegebenen Bild.
        height, width, _ = image.shape
        # Ignoriert den oberen Teil des Bildes, um nur relevante Bereiche zu analysieren.
        ignore_height = int(height * 0.31)
        image_cropped = image[:height - ignore_height, :]
        # Umwandlung des Bildes in Graustufen.
        gray = self.rgb_to_gray(image_cropped)
        # Anwendung des Sobel-Operators zur Kantenfindung.
        edges = self.detect_edges(gray)
        # Umwandlung der Kanten in ein Array von Ganzzahlen für die Verarbeitung.
        edge_array = self.edges_to_array(edges)
        # Extraktion der Startpunkte der Fahrspurmarkierungen.
        initial_points = self.extract_lane_edges(edge_array)
        # Analyse und Klassifizierung der Kantenpunkte für linke und rechte Fahrspuren.
        left_edges, right_edges = self.get_edge_coordinates(edge_array, initial_points)
        # Erstellung des Debug-Bildes mit initialen Markierungspunkten.
        self.debug_image = self.create_debug_image(edges, initial_points)
        # Einfärben des Debug-Bildes basierend auf den klassifizierten Kanten.
        self.spread_colors(left_edges, right_edges)
        # Rückgabe des bearbeiteten Debug-Bildes.
        return self.debug_image, left_edges, right_edges

    def rgb_to_gray(self, rgb):
        # Konvertierung eines RGB-Bildes in ein Graustufenbild.
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def detect_edges(self, gray: np.ndarray, threshold: int = 120):
        # Anwendung des Sobel-Operators in x- und y-Richtung und Kombination der Ergebnisse.
        sobel_x = sobel(gray, axis=1)
        sobel_y = sobel(gray, axis=0)
        # Berechnung der Gesamtkantenstärke und Binarisierung basierend auf einem Schwellenwert.
        edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edges = (edges > threshold) * 1
        return edges

    def edges_to_array(self, edges):
        # Konvertierung des Kantenbildes in ein Integer-Array für die weitere Verarbeitung.
        return edges.astype(np.int32)

    def extract_lane_edges(self, edge_array):
        # Bestimmt die untersten Punkte der linken und rechten Fahrspurmarkierungen.
        height, width = edge_array.shape
        mid = width // 2
        # Sucht nach den extremen Kantenpunkten am unteren Rand des Bildes.
        bottom_indices = np.where(edge_array[height - 1, :] == 1)[0]
        leftmost_index = max((index for index in bottom_indices if index < mid), default=None)
        rightmost_index = min((index for index in bottom_indices if index > mid), default=None)
        return [(height - 1, leftmost_index, 'left'), (height - 1, rightmost_index, 'right')]

    def get_edge_coordinates(self, edge_array, initial_points, min_distance=2):
        # Erzeugt Listen von Koordinaten für linke und rechte Kanten.
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
                    edge_array[ny, nx] = 0  # Markiert als besucht
                    if side == 'left':
                        # Überprüfen, ob ein Punkt auf der gleichen Höhe innerhalb des Mindestabstands ist
                        if not any(nx - min_distance <= existing_x <= nx + min_distance for existing_x in
                                   left_y_coords.get(ny, [])):
                            if ny not in left_y_coords:
                                left_y_coords[ny] = []
                            left_y_coords[ny].append(nx)
                            left_edges.append((nx, ny))
                    else:
                        # Überprüfen, ob ein Punkt auf der gleichen Höhe innerhalb des Mindestabstands ist
                        if not any(nx - min_distance <= existing_x <= nx + min_distance for existing_x in
                                   right_y_coords.get(ny, [])):
                            if ny not in right_y_coords:
                                right_y_coords[ny] = []
                            right_y_coords[ny].append(nx)
                            right_edges.append((nx, ny))
                    queue.append((nx, ny, side))
        return left_edges, right_edges

    def spread_colors(self, left_edges, right_edges):
        # Einfärben der Fahrspurkanten im Debug-Bild.
        for x, y in left_edges:
            self.debug_image[y, x] = [255, 0, 0]  # Rot für die linke Spur
        for x, y in right_edges:
            self.debug_image[y, x] = [0, 0, 255]  # Blau für die rechte Spur

    def create_debug_image(self, edges, initial_points):
        # Initialisiert das Debug-Bild und markiert die Startpunkte.
        debug_image = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for point in initial_points:
            y, x, side = point
            if x is not None:
                color = [255, 0, 0] if side == 'left' else [0, 0, 255]
                debug_image[y, x] = color
        return debug_image
