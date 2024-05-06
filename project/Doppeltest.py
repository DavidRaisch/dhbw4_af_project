from __future__ import annotations
import argparse
import cv2
import numpy as np
import gymnasium as gym

from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection
from path_planning import PathPlanning


def run(env, input_controller: InputController):
    # Initialisierung der LaneDetection- und PathPlanning-Klassen
    lane_detection = LaneDetection()
    path_planning = PathPlanning()

    # Seed für die Umgebung festlegen
    seed = int(np.random.randint(0, int(1e6)))
    print(seed)
    seed = 823827
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        # Schritt für Fahrspurerkennung
        edges = lane_detection.detect(state_image)
        # Linke und rechte Fahrspurkanten extrahieren
        left_lane_boundary, right_lane_boundary = extract_lane_boundaries(lane_detection.debug_image)

        # Sicherstellen, dass die Grenzen gültig sind
        if left_lane_boundary and right_lane_boundary:
            # Pfadplanung durchführen, basierend auf den erkannten Grenzen
            way_points, curvature = path_planning.plan(left_lane_boundary, right_lane_boundary)
        else:
            way_points, curvature = [], 0

        # Debug-Bild auf die gewünschte Größe skalieren
        debug_image_resized = cv2.resize(lane_detection.debug_image, (800, 600))  # Anpassen der Größe nach Bedarf

        # Wegpunkte proportional zur neuen Bildgröße skalieren und visualisieren
        if way_points:
            for point in way_points:
                if not np.isnan(point).any():  # Überprüfen auf NaN-Werte
                    scaled_point = (int(point[0] * 800 / lane_detection.debug_image.shape[1]), int(point[1] * 600 / lane_detection.debug_image.shape[0]))
                    cv2.circle(debug_image_resized, scaled_point, 1, (255, 255, 255), -1)

        # Debug-Bild anzeigen
        cv2.imshow('Car Racing - Lane Detection & Path Planning', debug_image_resized)
        cv2.waitKey(1)

        # Umgebung aktualisieren und Aktionen ausführen
        input_controller.update()
        a = [input_controller.steer, input_controller.accelerate, input_controller.brake]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Überprüfen, ob Episode beendet oder übersprungen werden soll
        if done or input_controller.skip:
            print(f"seed: {seed:06d}     reward: {total_reward:06.2F}")
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0


def extract_lane_boundaries(debug_image):
    """Extrahiert die Fahrspurgrenzen aus dem Debug-Bild, das von LaneDetection generiert wurde."""
    left_lane_boundary = []
    right_lane_boundary = []
    for y, row in enumerate(debug_image):
        left_indices = np.where((row[:, 0] == 255) & (row[:, 1] == 0) & (row[:, 2] == 0))[0]  # Rote Punkte
        right_indices = np.where((row[:, 0] == 0) & (row[:, 1] == 0) & (row[:, 2] == 255))[0]  # Blaue Punkte
        if left_indices.size > 0:
            left_lane_boundary.append((left_indices[0], y))
        if right_indices.size > 0:
            right_lane_boundary.append((right_indices[0], y))
    return left_lane_boundary, right_lane_boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = CarRacingEnvWrapper(gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False))
    input_controller = InputController()

    run(env, input_controller)
    env.close()


if __name__ == '__main__':
    main()
