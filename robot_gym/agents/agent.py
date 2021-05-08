from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, env_id, env_args, num_agents, log_dir, debug_mode):
        """
        A generic agent instance.
        :param env_args: The Gym environment arguments.
        :param num_agents: The number of parallel agents.
        :param debug_mode: Boolean. True for 1 agent.
        :param log_dir: The directory used to save the trained policy.
        """
        self._env_id = env_id
        self._env_args = env_args
        self._num_agent = num_agents
        self._debug = debug_mode
        self._log_dir = log_dir

    @abstractmethod
    def train(self):
        pass
