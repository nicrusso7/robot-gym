from robot_gym.agents.ddpg.ddpg import DDPG
from robot_gym.agents.ppo.ppo import PPO
from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.rex import rex


CONTROLLERS = {
    'mpc': mpc_controller.MPCController
}

ROBOTS = {
    'rex': rex.Rex
}

AGENTS = {
    'ppo': PPO,
    'ddpg': DDPG
}
