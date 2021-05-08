import numpy as np

from robot_gym.model.robots import simple_motor

# -------------------------------------------------
# Pose Controller
# -------------------------------------------------
MOTOR_CONTROL_MODE = simple_motor.MOTOR_CONTROL_POSITION

l = 0.23
w = 0.075
hip = 0.055
leg = 0.10652
foot = 0.145
y_dist = 0.185
x_dist = l
height = 0.2
# frame vectors
hip_front_right_v = np.array([l / 2, -w / 2, 0])
hip_front_left_v = np.array([l / 2, w / 2, 0])
hip_rear_right_v = np.array([-l / 2, -w / 2, 0])
hip_rear_left_v = np.array([-l / 2, w / 2, 0])
foot_front_right_v = np.array([x_dist / 2, -y_dist / 2, -height])
foot_front_left_v = np.array([x_dist / 2, y_dist / 2, -height])
foot_rear_right_v = np.array([-x_dist / 2, -y_dist / 2, -height])
foot_rear_left_v = np.array([-x_dist / 2, y_dist / 2, -height])
