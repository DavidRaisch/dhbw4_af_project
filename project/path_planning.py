import numpy as np

class PathPlanning:
    def __init__(self):
        # Konstruktor der Klasse PathPlanning ohne spezifische Attribute.
        pass

    def plan(self, left_lane_points, right_lane_points):
        """
        Berechnet die Mittellinie und die Krümmung basierend auf den linken und rechten Spurbegrenzungen.

        :param left_lane_points: Koordinaten der linken Spurbegrenzung
        :param right_lane_points: Koordinaten der rechten Spurbegrenzung
        :return: Liste der Mittellinienpunkte, reduziert auf jeden 10. Punkt, und die durchschnittliche Krümmung
        """
        # Umwandlung der Eingabelisten in NumPy Arrays
        left_lane_points = np.array(left_lane_points)
        right_lane_points = np.array(right_lane_points)

        # Berechnung der Mittelpunkte zwischen den linken und rechten Punkten
        midpoints = self.calculate_midpoints(left_lane_points, right_lane_points)

        # Berechnung der Krümmung basierend auf diesen Mittelpunkten
        curvature = self.calculate_curvature(midpoints)


        #Filtern von Fehlerhaften Berechnungen
        if curvature > 25:
            curvature = 25

        # Auswahl jedes 10. Punktes der Mittellinie für die Ausgabe
        sampled_midpoints = midpoints[::10]

        return sampled_midpoints, curvature

    def calculate_midpoints(self, left_lane_points, right_lane_points):
        """
        Berechnet die Mittelpunkte der Spur.

        :param left_lane_points: Array der Koordinaten der linken Spurbegrenzung
        :param right_lane_points: Array der Koordinaten der rechten Spurbegrenzung
        :return: Array der berechneten Mittelpunkte
        """
        if not left_lane_points.any() or not right_lane_points.any():
            # Gibt ein leeres Array zurück, falls eines der Eingangsarrays leer ist
            return np.array([])

        # Bestimmung der kürzeren Länge der beiden Punktlisten
        min_len = min(len(left_lane_points), len(right_lane_points))

        # Extraktion der x- und y-Koordinaten
        left_x, left_y = left_lane_points[:min_len].T
        right_x, right_y = right_lane_points[:min_len].T

        # Berechnung der Mittelpunkte
        mid_x = (left_x + right_x) / 2
        mid_y = (left_y + right_y) / 2
        midpoints = np.column_stack((mid_x, mid_y))

        return midpoints

    def calculate_curvature(self, points):
        """
        Berechnet die Krümmung der Punkte auf der Mittellinie nach Voranalyse, die nur die durchgehende Linie berücksichtigt,
        die von der untersten Stelle (höchster y-Wert) beginnend nach oben verläuft ohne Unterbrechungen von mehr als einem Pixel.

        :param points: Array der Mittelpunkte
        :return: Durchschnittliche Krümmung
        """
        try:
            if points.size == 0:
                # Wenn keine Punkte vorhanden sind, ist die Krümmung 0
                return 0

            # Sortieren der Punkte nach der y-Koordinate von unten nach oben (absteigend sortieren)
            points = points[np.argsort(-points[:, 1])]

            # Filtern der Punkte, um sicherzustellen, dass nur die unterste Linie verwendet wird
            filtered_points = [points[0]]
            for i in range(1, len(points)):
                # Nur die Punkte hinzufügen, die höchstens 1 Pixel vertikal und 5 Pixel horizontal von dem vorherigen Punkt entfernt sind
                if abs(points[i][1] - filtered_points[-1][1]) <= 1 and abs(points[i][0] - filtered_points[-1][0]) <= 5:
                    filtered_points.append(points[i])
                elif points[i][1] < filtered_points[-1][1] - 1:
                    break  # Unterbrechung der Linie
            filtered_points = np.array(filtered_points)

            if filtered_points.size == 0:
                return 0

            # Berechnung der ersten und zweiten Ableitungen
            dx_dt = np.gradient(filtered_points[:, 0])
            dy_dt = np.gradient(filtered_points[:, 1])
            d2x_dt2 = np.gradient(dx_dt)
            d2y_dt2 = np.gradient(dy_dt)

            # Sicherstellen, dass der Nenner nicht zu klein wird
            denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
            safe_denominator = np.where(denominator < 1e-6, 1e-6, denominator)

            # Berechnung der Krümmung
            curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / safe_denominator
            curvature = curvature * 10# Skalierung der Krümmung für bessere Lesbarkeit

            return curvature.mean()
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0



