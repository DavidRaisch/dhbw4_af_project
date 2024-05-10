from __future__ import annotations

import numpy as np
import time

class LongitudinalControl:
    def __init__(self):
        self.kp = 0.1
        self.ki = 0.000001
        self.kd = 0.001
        self.error_integral = 0
        self.prev_error = 0
        self.start_time = time.time()

    def control(self, current_speed: float, target_speed: float, steering_angle: float) -> tuple[float, float]:
    # Hier wird die Regelung der Beschleunigung und des Bremsens durchgeführt
        error = target_speed - current_speed
        self.error_integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        control_signal = self.kp * error  + self.kd * derivative #+ self.ki * self.error_integral
        #print(control_signal)
        # Berechnung von Beschleunigung und Bremsen basierend auf dem Steuersignal
       
        if control_signal > 0:
            acceleration = min(control_signal, 1.0)  # Maximalwert auf 1.0 begrenzen
            brake = 0.0
        else:
            acceleration = 0.0
            brake = min(-control_signal, 1.0)  # Maximalwert auf 1.0 begrenzen

        

        return acceleration, brake

    def predict_target_speed(self, curvature, steeringangle):
        # Hier wird die Zielgeschwindigkeit anhand der Kurvenkrümmung bestimmt
        # Ein einfaches Beispiel: niedrigere Geschwindigkeit bei hoher Kurvenkrümmung
        # Hier können Sie eine ausgefeiltere Logik einfügen, z. B. mit weiteren Heuristiken
        # In diesem Beispiel wird die Zielgeschwindigkeit direkt aus der Kurvenkrümmung abgeleitet
        curv = curvature
        #print(steering_angle)
        #target_speed = max(30.0, 20.0 - curv * 1.0)
        #print(curv)
       
        timer = time.time() - self.start_time
        #print(timer)
        if timer < 1:
            target_speed = 20.0
        else:
            #if curv > 30:
                #target_speed = 20
            if curv > 50: 
                target_speed = 50.0 - curv*0.3
            elif curv > 20:
                target_speed = 50.0 - curv*0.4
            elif curv < 20:
                target_speed = 50 - curv*0.5
            else:
                target_speed = 25

        if abs(steeringangle) > 0.8:
            target_speed = 10
            print("Regelung Failed")
        #print(curv)
        #print(target_speed)
        return target_speed


    def _calculate_curvature(self, trajectory: list[tuple[float, float]]) -> float:
        # Berechnung der Kurvenkrümmung basierend auf den Trajektoriepunkten
        # Wir verwenden hier eine einfache Methode: numerische Berechnung der Krümmung
        # durch Anpassung eines Kreises an die Trajektoriepunkte in der Umgebung eines Punktes
        x = np.array([p[0] for p in trajectory])
        y = np.array([p[1] for p in trajectory])
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5)

        curvature = abs(curvature) * 1000
        return np.mean(curvature)

