import re
import numpy as np

NUM_LEG = 4
START_POS = [0, 0, 0.48]
INIT_ORIENTATION = [0, 0, 0]

DEFAULT_ABDUCTION_ANGLE = 0
DEFAULT_HIP_ANGLE = 0.67
DEFAULT_KNEE_ANGLE = -1.25

ARM_REST_MOTOR_ANGLES = np.array([
        -1.57, 0., 0.,
        0., 0., 0.
    ])

LEG_STAND_MOTOR_ANGLES = np.array(
    [
        DEFAULT_ABDUCTION_ANGLE,
        DEFAULT_HIP_ANGLE,
        DEFAULT_KNEE_ANGLE
    ] * 4)

INIT_MOTOR_ANGLES = np.concatenate([LEG_STAND_MOTOR_ANGLES, ARM_REST_MOTOR_ANGLES])

IDENTITY_ORIENTATION = [0, 0, 0, 1]

CHASSIS_NAME_PATTERN = re.compile(r"\w+body\w+")
HIP_NAME_PATTERN = re.compile(r"\w+_to_abduct_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_to_thigh_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_to_knee_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
ARM_NAME_PATTERN = re.compile(r"\w+_arm_\w+")

HIP_JOINTS_PATTERN = "to_abduct"
UPPER_JOINTS_PATTERN = "to_thigh"
LOWER_JOINTS_PATTERN = "to_knee"
ARM_JOINTS_PATTERN = "motor_arm"

DEFAULT_HIP_POSITIONS = (
    (0.22, -0.105, 0),
    (0.22, 0.105, 0),
    (-0.22, -0.105, 0),
    (-0.22, 0.105, 0),
)

BODY_B_FIELD_NUMBER = 2
LINK_A_FIELD_NUMBER = 3

HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
LOWER_LEG_JOINT_OFFSET = 0.0
