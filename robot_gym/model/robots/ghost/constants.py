import re
import numpy as np

NUM_LEG = 4
START_POS = [0, 0, 0.48]
INIT_ORIENTATION = [0, 0, 0]

DEFAULT_ABDUCTION_ANGLE = 0
DEFAULT_HIP_ANGLE = 0.67
DEFAULT_KNEE_ANGLE = -1.25

INIT_MOTOR_ANGLES = np.array(
    [
        DEFAULT_ABDUCTION_ANGLE,
        DEFAULT_HIP_ANGLE,
        DEFAULT_KNEE_ANGLE
    ] * 4)

IDENTITY_ORIENTATION = [0, 0, 0, 1]

CHASSIS_NAME_PATTERN = re.compile(r"\w+body\w+")
HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")

HIP_JOINTS_PATTERN = "hip_joint"
UPPER_JOINTS_PATTERN = "upper_joint"
LOWER_JOINTS_PATTERN = "lower_joint"

DEFAULT_HIP_POSITIONS = (
    (0.22, -0.1, 0),
    (0.22, 0.1, 0),
    (-0.22, -0.1, 0),
    (-0.22, 0.1, 0),
)

BODY_B_FIELD_NUMBER = 2
LINK_A_FIELD_NUMBER = 3

HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
LOWER_LEG_JOINT_OFFSET = 0.0
