r"""Running a pre-trained agent"""
import logging
import os
import site
import time

import tensorflow.compat.v1 as tf

from robot_gym.agents.ppo import simple_ppo_agent
from robot_gym.agents.ppo.scripts import utility
from robot_gym.util.cli import flags


class PolicyPlayer:
    def __init__(self, env_id: str, controller_id: str, ctrl_obj: object, robot: str, args: dict):
        self.gym_dir_path = str(site.getsitepackages()[-1])
        self._env_id = env_id
        self._ctrl_id = controller_id
        self._args = args
        self._args['robot_model'] = robot
        self._args['controller_class'] = ctrl_obj
        self._args['debug'] = True
        self._args['policy'] = True
        self._args['render'] = True

    def play(self):
        policy_id = f"{self._env_id}_{self._ctrl_id}"
        policy_path = flags.ENV_ID_TO_POLICY[policy_id][0]
        policy_dir = os.path.join(self.gym_dir_path, policy_path)
        config = utility.load_config(policy_dir)
        policy_layers = config.policy_layers
        value_layers = config.value_layers
        env = config.env(**self._args)
        network = config.network
        checkpoint = os.path.join(policy_dir, flags.ENV_ID_TO_POLICY[policy_id][1])
        with tf.Session() as sess:
            agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                                     env,
                                                     network,
                                                     policy_layers=policy_layers,
                                                     value_layers=value_layers,
                                                     checkpoint=checkpoint)
            sum_reward = 0
            observation = env.reset()
            while True:
                action = agent.get_action([observation])
                observation, reward, done, _ = env.step(action[0])
                time.sleep(0.002)
                sum_reward += reward
                logging.info(f"Reward={sum_reward}")
                if done:
                    break
