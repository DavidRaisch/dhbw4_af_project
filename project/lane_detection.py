import numpy as np
import cv2

class LaneDetection:
    def __init__(self, num_points=7):
        self.canny_low_threshold = 30
        self.canny_high_threshold = 90
        self.gaussian_blur_kernel_size = (5, 5)
        self.ignore_bottom_fraction = 0.32
        self.num_points = num_points

    def detect(self, image: np.ndarray) -> np.ndarray:
        height = image.shape[0]
        ignore_height = int(height * self.ignore_bottom_fraction)
        image = image[:height - ignore_height, :]

        s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(s_channel, self.gaussian_blur_kernel_size, 0)
        edges = cv2.Canny(blur, self.canny_low_threshold, self.canny_high_threshold)
        self.debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(self.debug_image, contours, -1, (0, 255, 0), 2)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Betrachte nur die zwei größten Konturen
            contours = sorted(contours, key=lambda c: np.mean([p[0][0] for p in c]))  # Sortiere nach horizontaler Position

            points = self.collect_contour_points(contours)

            for point_set, color in zip(points, [(255, 0, 0), (0, 0, 255)]):
                for point in point_set:
                    cv2.circle(self.debug_image, point, radius=2, color=color, thickness=-1)

        return edges, self.debug_image

    def collect_contour_points(self, contours):
        points = []
        for contour in contours:
            if len(contour) > 1:
                curve = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
                curve = np.array([p[0] for p in curve], dtype=np.float32)
                dist = np.linspace(0, 1, self.num_points)
                uniform_points = np.array([self.interpolate_line(curve, t) for t in dist])
                points.append([tuple(map(int, pt)) for pt in uniform_points])
            else:
                # Falls Kontur zu kurz ist, verwende den einzigen Punkt mehrmals
                repeated_point = tuple(contour[0][0])
                points.append([repeated_point for _ in range(self.num_points)])
        return points

    def interpolate_line(self, points, t):
        """ Linear interpolation of points along a simplified contour. """
        if t == 1:
            return points[-1]
        num = len(points) - 1
        idx = int(num * t)
        pt1, pt2 = points[idx], points[idx + 1]
        alpha = num * t - idx
        return (1 - alpha) * pt1 + alpha * pt2
