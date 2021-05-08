import datetime
import functools
import logging
import os
import platform

import gym
import tensorflow.compat.v1 as tf

from robot_gym.agents.agent import Agent
from robot_gym.agents.ppo.scripts import utility, configs
from robot_gym.agents.ppo.tools import wrappers
from robot_gym.agents.ppo.tools.attr_dict import AttrDict
from robot_gym.agents.ppo.tools.loop import Loop


class PPO(Agent):

    def _create_environment(self, config):
        """
        Constructor for an instance of the environment.

        Args:
          config: Object providing configurations via attributes.

        Returns:
          Wrapped OpenAI Gym environment.
        """
        env = gym.make(config.env, **self._env_args)
        if config.max_length:
            env = wrappers.LimitDuration(env, config.max_length)
        env = wrappers.RangeNormalize(env)
        env = wrappers.ClipAction(env)
        env = wrappers.ConvertTo32Bit(env)
        return env

    @staticmethod
    def _define_loop(graph, logdir, train_steps, eval_steps):
        """Create and configure a training loop with training and evaluation phases.

          Args:
            graph: Object providing graph elements via attributes.
            logdir: Log directory for storing checkpoints and summaries.
            train_steps: Number of training steps per epoch.
            eval_steps: Number of evaluation steps per epoch.

          Returns:
            Loop object.
        """
        loop = Loop(logdir, graph.step, graph.should_log, graph.do_report, graph.force_reset)
        loop.add_phase('train',
                       graph.done,
                       graph.score,
                       graph.summary,
                       train_steps,
                       report_every=None,
                       log_every=train_steps // 2,
                       checkpoint_every=None,
                       feed={graph.is_training: True})
        loop.add_phase('eval',
                       graph.done,
                       graph.score,
                       graph.summary,
                       eval_steps,
                       report_every=eval_steps,
                       log_every=eval_steps // 2,
                       checkpoint_every=10 * eval_steps,
                       feed={graph.is_training: False})
        return loop

    def _start_training(self, config, env_processes):
        """Training and evaluation entry point yielding scores.

          Resolves some configuration attributes, creates environments, graph, and
          training loop. By default, assigns all operations to the CPU.

          Args:
            config: Object providing configurations via attributes.
            env_processes: Whether to step environments in separate processes.

          Yields:
            Evaluation scores.
        """
        tf.reset_default_graph()
        with config.unlocked:
            config.network = functools.partial(utility.define_network, config.network, config)
            config.policy_optimizer = getattr(tf.train, config.policy_optimizer)
            config.value_optimizer = getattr(tf.train, config.value_optimizer)
        if config.update_every % config.num_agents:
            logging.warning('Number of _agents should divide episodes per update.')
        with tf.device('/cpu:0'):
            agents = self._num_agent if self._num_agent is not None else config.num_agents
            num_agents = 1 if self._debug else agents
            batch_env = utility.define_batch_env(lambda: self._create_environment(config), num_agents, env_processes)
            graph = utility.define_simulation_graph(batch_env, config.algorithm, config)
            loop = self._define_loop(graph, config.logdir, config.update_every * config.max_length,
                                     config.eval_episodes * config.max_length)
            total_steps = int(config.steps / config.update_every * (config.update_every + config.eval_episodes))
        # Exclude episode related variables since the Python state of environments is
        # not checkpointed and thus new episodes start after resuming.
        saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            utility.initialize_variables(sess, saver, config.logdir)
            for score in loop.run(sess, saver, total_steps):
                yield score
        batch_env.close()

    def train(self):
        """Create configuration and launch the trainer."""
        utility.set_up_logging()
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        full_logdir = os.path.expanduser(os.path.join(self._log_dir, '{}-{}'.format(timestamp, self._env_id)))
        config = AttrDict(getattr(configs, self._env_id)())
        config = utility.save_config(config, full_logdir)
        os_name = platform.system()
        enable_processes = False if os_name == 'Windows' else True
        for score in self._start_training(config, enable_processes):
            tf.logging.info('Score {}.'.format(score))
