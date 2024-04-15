from __future__ import annotations
class LongitudinalControl:
    def __init__(self, kp=0.1, ki=0.01, kd=0.005):
        # Initialisierung der PID-Koeffizienten
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def control(self, current_speed, target_speed, steer_angle, dt=1):
        # Berechnung des Fehlers
        error = target_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error

        # Anpassen der Beschleunigung/Bremsen basierend auf PID-Output
        acceleration = max(pid_output, 0)  # Positive PID-Ausgaben führen zur Beschleunigung
        braking = -min(pid_output, 0)  # Negative PID-Ausgaben führen zum Bremsen

        return acceleration, braking

    def predict_target_speed(self, trajectory_info, current_speed, steer_angle):
        # Beispielhafte Berechnung der Zielgeschwindigkeit
        road_curvature = abs(steer_angle)  # Vereinfachte Annahme zur Krümmungsberechnung
        if road_curvature > 0.1:
            target_speed = max(30, 100 - road_curvature * 50)
        else:
            target_speed = 100  # Höhere Zielgeschwindigkeit bei geringer Krümmung
        return target_speed
