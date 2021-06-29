from robot_gym.agents.agent import Agent
from robot_gym.agents.stanford.sac import simple_sac_agent


class SAC(Agent):

    def train(self):
        simple_sac_agent.train(log_dir=self._log_dir,
                               env_class=self._env_args["env_class"],
                               robot_model=self._env_args["robot_model"],
                               mark=self._env_args["mark"],
                               controller_class=self._env_args["controller_class"],
                               config=self._env_args["config"],
                               render=self._env_args["render"],
                               debug=self._debug,
                               policy=self._debug)
