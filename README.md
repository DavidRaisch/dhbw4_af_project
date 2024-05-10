# Autonomous Driving Project
Here is the project folder description by:
- David Raisch (Matrikelnummer)
- Mattes Wirths (Matrikelnummer)

## Table of Contents
- [Introduction](#introduction)
- [Lateral Control Algorithm](#lateral-control-algorithm)
- [Longitudinal Control](#longitudinal-control)
- [Extensions](#extensions)
- [Potential Bonus Points](#potential-bonus-points)

## Introduction

This project folder contains various modules and functionalities aimed at achieving autonomous driving.

## Lateral Control

This file contains a lateral control for determining vehicle steering angles using either pure pursuit or Stanley methods.

### Description

- **Purpose:** Controls vehicle steering based on trajectory tracking.
- **Implementation:** Implemented in Python using `numpy` for numerical operations.
- **Operation:**
  - Pure Pursuit: Determines steering angle based on the angle between the lookahead point and the current vehicle position.
  - Stanley: Calculates cross-track error and heading error to determine the steering angle.
  
Pure Pursuit is the better implemented controller and should be utilized. Stanley, which operates only with info['trajectory'], needs to be passed as an argument to the control() function in car.py.


## Longitudinal Control

This file implements a longitudinal control for vehicle speed regulation based on PID control and curvature-based speed prediction.

### Description

- **Purpose:** Controls vehicle speed longitudinally using PID control and predicts target speed based on road curvature.
- **Implementation:** Utilizes Python with `numpy` for numerical operations and `time` for timing functionalities.
- **Operation:** Calculates acceleration/braking via `control()` method and predicts target speed via `predict_target_speed()



## Extensions
- **Randomization feature in lane detection:** This feature introduces randomization in the lane detection process to enhance robustness against varying environmental conditions and improve generalization. !!!!Anpassen!!!!
- **Implementation and functionality of both controllers:** 
Details regarding their implementation and functionality are provided in the code documentation and the "Lateral Control" chapter.

## Potential Bonus Points
- **Custom test scripts:**
  - *Doppeltest.py:* This script tests the integration of Path Planning and Lane Detection functionalities.
  - *Trippletest.py:* Tests the combined functionality of Path Planning, Lane Detection, and Lateral Control modules.
- **High average score:** Aiming for a high average score across all testing scenarios to ensure robust performance.




