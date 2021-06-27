import gym
from gibson2.render.profiler import Profiler

import numpy as np

from robot_gym.controllers.mpc import mpc_controller

from robot_gym.gym.gibson_gym_env import GibsonGymEnv

from robot_gym.model.robots.ghost import ghost


class ReachEnv(GibsonGymEnv):

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 config,
                 debug=False,
                 policy=False,
                 render=False,
                 ):

        super(ReachEnv, self).__init__(robot_model, mark, controller_class, config, debug, policy, render)

    def setup_action_space(self):
        action_high = 0.5
        action_dim = 2
        self._base_sim.robot.action_high = np.array([action_high] * action_dim)
        self._base_sim.robot.action_low = -self._base_sim.robot.action_high
        self._base_sim.robot.action_space = gym.spaces.Box(shape=(action_dim,),
                                                           low=-action_high,
                                                           high=action_high,
                                                           dtype=np.float32)


def main():
    env = ReachEnv(ghost.Ghost, "1", mpc_controller.MPCController, "reach.yaml", debug=True, render=True)
    for j in range(100):
        env.reset()
        for i in range(10000):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                # if done:
                #     logging.info(
                #         "Episode finished after {} timesteps".format(i + 1))
                #     break
    env.close()


if __name__ == "__main__":
    main()
