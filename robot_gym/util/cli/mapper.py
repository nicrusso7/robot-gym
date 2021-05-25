from robot_gym.agents.ddpg.ddpg import DDPG
from robot_gym.agents.ppo.ppo import PPO
from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.ghost import ghost
from robot_gym.model.robots.k3lso import k3lso

CONTROLLERS = {
    'mpc': mpc_controller.MPCController
}

ROBOTS = {
    'ghost': ghost.Ghost,
    'k3lso': k3lso.K3lso
}

AGENTS = {
    'ppo': PPO,
    'ddpg': DDPG
}
