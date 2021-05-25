import numpy as np

from robot_gym.model.robots import simple_motor

NUM_MOTORS = 12

MOTOR_ENABLED = [True] * NUM_MOTORS

MOTOR_OFFSET = np.array([0.] * NUM_MOTORS)

MOTOR_DIRECTION = np.array([1] * NUM_MOTORS)

MOTOR_POSITION_GAINS = [220.] * NUM_MOTORS

MOTOR_VELOCITY_GAINS = np.array([1., 2., 2.] * 4)

MOTOR_CONTROL_CLASS = simple_motor.RobotMotorModel

MOTOR_MODEL = simple_motor
