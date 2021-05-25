from mpc_controller import gait_generator as gait_generator_lib

from robot_gym.model.robots import simple_motor

# -------------------------------------------------
# MPC Controller
# -------------------------------------------------
MPC_BODY_MASS = 190 / 9.8
MPC_BODY_INERTIA = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447)
MPC_BODY_HEIGHT = 0.38
MPC_VELOCITY_MULTIPLIER = 1.0

STANCE_DURATION_SECONDS = [0.3] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).
MAX_TIME_SECONDS = 30.

# Standing
# DUTY_FACTOR = [1.] * 4
# INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]
#
# INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Trotting
DUTY_FACTOR = [0.6] * 4
INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

# VX_OFFSET = 0.
# VY_OFFSET = 0.08
# WZ_OFFSET = -0.025
VX_OFFSET = 0.
VY_OFFSET = 0.
WZ_OFFSET = 0.
