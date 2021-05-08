import os
import datetime

import gym

from robot_gym.agents.agent import Agent
from robot_gym.agents.ddpg import simple_ddpg_agent, constants


class DDPG(Agent):

    def train(self):
        env = gym.make("GoTo-v0", **self._env_args)
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        full_logdir = os.path.expanduser(os.path.join(self._log_dir, '{}-{}'.format(timestamp, self._env_id)))
        simple_ddpg_agent.train(env=env, dir_name=full_logdir, steps=constants.MAX_STEPS)
