import numpy as np

class PathPlanning:
    def __init__(self):
        pass

    def plan(self, left_lane_points, right_lane_points):
        # Konvertiere Listen in NumPy Arrays
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)
        # Berechne die Mittelpunkte der Fahrspuren
        midpoints = self.calculate_midpoints(left_lane_points, right_lane_points)
        # Berechne die Krümmung der Mittelpunkte
        curvature = self.calculate_curvature(midpoints)
        # Berechne die Ideallinie basierend auf den Mittelpunkten und der Krümmung
        ideal_line = self.calculate_ideal_line(midpoints, curvature)
        return ideal_line.tolist(), curvature

    def calculate_midpoints(self, left_lane_points, right_lane_points):
        # Bestimme die kleinere Länge der beiden Punktelisten
        min_len = min(len(left_lane_points), len(right_lane_points))
        # Extrahiere x- und y-Koordinaten
        left_x, left_y = left_lane_points[:min_len].T
        right_x, right_y = right_lane_points[:min_len].T
        # Berechne die x- und y-Koordinaten der Mittelpunkte
        mid_x = (left_x + right_x) / 2
        mid_y = (left_y + right_y) / 2
        # Kombiniere die Koordinaten zu Mittelpunkten
        midpoints = np.column_stack((mid_x, mid_y))
        return midpoints

    def calculate_curvature(self, points):
        # Berechne die erste Ableitung der x- und y-Koordinaten
        dx_dt = np.gradient(points[:, 0])
        dy_dt = np.gradient(points[:, 1])
        # Berechne die zweite Ableitung der x- und y-Koordinaten
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        # Berechne die Krümmung der Punkte
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
        return curvature.mean()

    def calculate_ideal_line(self, midpoints, curvature):
        # Bestimme die Richtung der Krümmung
        if curvature > 0:
            direction = 1
        else:
            direction = -1
        # Führe eine Parabelfit für die y-Koordinaten der Mittelpunkte durch
        x = np.linspace(0, 1, len(midpoints))
        y = midpoints[:, 1]
        coef = np.polyfit(x, y, 2)
        # Verstärke den Einfluss der Krümmung signifikant
        curvature_adjustment = direction * curvature * 500  # Erhöhter Multiplikator für deutlicheren Effekt
        adjusted_y = coef[0] * (x**2) + coef[1] * x + coef[2] + curvature_adjustment
        ideal_line = np.column_stack((midpoints[:, 0], adjusted_y))
        return ideal_line
