# Autonomous Driving Project
Here is the project folder description by:
- David Raisch (3768392)
- Mattes Wirths (1788761)

## Table of Contents
- [Introduction](#introduction)
- [Lane Detection](#lane-detection)
- [Path Planning](#path-planning)
- [Lateral Control](#lateral-control)
- [Longitudinal Control](#longitudinal-control)
- [Extensions](#extensions)
- [Potential Bonus Points](#potential-bonus-points)
- [Improvments](#improvements)

## Introduction

This project folder contains various modules and functionalities aimed at achieving autonomous driving.

## Lane Detection

This file is using the input Image and analyzes it for the borders of the path.

### Description

- **Purpose:** Detects Borders of the Path based on the Input
- **Implementation:** Implemented in Python with image processing using `numpy` and `scipy`.
- **Operation:** detect: Generates a Gray-Scaling-Image, recognizes Borders and checks if left or right in detect method.
	

## Path Planning

This file generates a Path based on the Borders of the detected Borders.

### Description

- **Purpose:** Plans a Path for the vehicle to follow. 
- **Implementation:** Implemented in Python using `numpy`. 
- **Operation:** plan: Uses the Edges of the path to calculate the Points in the Middle of the Road. These are used to calculate the curvature.

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
- **Randomization feature in lane detection:** The Detection of Borders isn't based on fixed colours. It is based on differences in color. Therefore our Implementation is robust against random colours for the road and environment as long as the difference of them is high enough.
- **Implementation and functionality of both controllers:** 
Details regarding their implementation and functionality are provided in the code documentation and the "Lateral Control" chapter.

## Potential Bonus Points
- **Custom test scripts:**
  - *Doppeltest.py:* This script tests the integration of Path Planning and Lane Detection functionalities.
  - *Trippletest.py:* Tests the combined functionality of Path Planning, Lane Detection, and Lateral Control modules.
- **High average score:** Aiming for a high average score across all testing scenarios to ensure robust performance.

## Improvements:

If an issue occurs or the average scores are unusually low, please restart Main.py to get a better impression of the project's real performance.
  




