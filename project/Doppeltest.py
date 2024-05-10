from __future__ import annotations

import argparse
import gymnasium as gym
import cv2
import numpy as np

from env_wrapper import CarRacingEnvWrapper
from input_controller import InputController
from lane_detection import LaneDetection
from path_planning import PathPlanning


def run(env, input_controller: InputController):
    lane_detection = LaneDetection()
    path_planning = PathPlanning()

    seed = int(np.random.randint(0, int(1e6)))
    state_image, info = env.reset(seed=seed)
    total_reward = 0.0

    while not input_controller.quit:
        # Detektion der Fahrspuren
        debug_image, left_edges, right_edges = lane_detection.detect(state_image)

        # Pfadplanung basierend auf den detektierten Fahrspuren
        sampled_midpoints, curvature = path_planning.plan(left_edges, right_edges)

        if np.isnan(sampled_midpoints).any():
            print("Warnung: NaN-Werte in Wegpunkten erkannt. Überspringe diese Iteration.")
            continue

        # Visualisierung vorbereiten
        cv_image = np.asarray(debug_image, dtype=np.uint8)
        sampled_midpoints = np.array(sampled_midpoints, dtype=np.int32)
        for point in sampled_midpoints:
            if 0 < point[0] < cv_image.shape[1] and 0 < point[1] < cv_image.shape[0]:
                cv_image[int(point[1]), int(point[0])] = [255, 255, 255]

        cv2.putText(cv_image, f"Krümmung: {curvature:.2f}", (15, 63), cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (cv_image.shape[1] * 6, cv_image.shape[0] * 6))
        cv2.imshow('Car Racing - Kombinierter Test', cv_image)
        cv2.waitKey(1)

        # Aktualisierung der Eingaben und Simulationsschritte
        input_controller.update()
        a = [input_controller.steer, input_controller.accelerate, input_controller.brake]
        state_image, r, done, trunc, info = env.step(a)
        total_reward += r

        if done or input_controller.skip:
            print(f"Seed: {seed:06d}     Gesamtpunktzahl: {total_reward:06.2F}")
            input_controller.skip = False
            seed = int(np.random.randint(0, int(1e6)))
            state_image, info = env.reset(seed=seed)
            total_reward = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", action="store_true", default=False)
    args = parser.parse_args()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = CarRacingEnvWrapper(gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False))
    input_controller = InputController()

    run(env, input_controller)
    env.reset()


if __name__ == '__main__':
    main()
