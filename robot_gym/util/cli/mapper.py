from robot_gym.agents.ddpg.ddpg import DDPG
from robot_gym.agents.ppo.ppo import PPO
from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.ghost import ghost


CONTROLLERS = {
    'mpc': mpc_controller.MPCController
}

ROBOTS = {
    'ghost': ghost.Ghost
}

AGENTS = {
    'ppo': PPO,
    'ddpg': DDPG
}
