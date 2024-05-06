from __future__ import annotations
import argparse
import cv2
import numpy as np
import gymnasium as gym
import pygame
from env_wrapper import CarRacingEnvWrapper
from lane_detection import LaneDetection
from path_planning import PathPlanning
from lateral_control import LateralControl

# Tastatur-Controller Klasse
class InputController:
    def __init__(self):
        pygame.init()
        self.quit = False
        self.skip = False
        self.accelerate = 0.0
        self.brake = 0.0

    def update(self):
        # Tastenaktionen festlegen
        keys = pygame.key.get_pressed()
        self.accelerate = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
        self.brake = 0.8 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0

        # Programm beenden und Simulation neu starten
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.skip = True

def run(env, input_controller: InputController):
    # Klasseninitialisierung
    lane_detection = LaneDetection()
    path_planning = PathPlanning()
    lateral_control = LateralControl()

    # Umgebung initialisieren
    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        input_controller.update()  # Tastatureingaben aktualisieren

        # Fahrspurerkennung
        edges = lane_detection.detect(state_image)
        left_lane_boundary, right_lane_boundary = extract_lane_boundaries(lane_detection.debug_image)

        # Pfadplanung
        if left_lane_boundary and right_lane_boundary:
            way_points, curvature = path_planning.plan(left_lane_boundary, right_lane_boundary)
        else:
            way_points, curvature = [], 0

        # Querregelung
        steering_angle = lateral_control.control(way_points, info['speed'])

        # Bild anzeigen und Skalierung
        debug_image_resized = cv2.resize(lane_detection.debug_image, (800, 600))
        if way_points:
            for point in way_points:
                if not np.isnan(point).any():
                    scaled_point = (int(point[0] * 800 / lane_detection.debug_image.shape[1]),
                                    int(point[1] * 600 / lane_detection.debug_image.shape[0]))
                    cv2.circle(debug_image_resized, scaled_point, 1, (255, 255, 255), -1)

        cv2.imshow('Car Racing - Combined System', debug_image_resized)
        cv2.waitKey(1)

        # Aktionen ausführen und Umgebung aktualisieren
        a = [steering_angle, input_controller.accelerate, input_controller.brake]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        # Überprüfen, ob die Episode beendet oder übersprungen werden soll
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
