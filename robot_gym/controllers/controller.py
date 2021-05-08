from abc import ABC, abstractmethod


class Controller(ABC):

    def __init__(self, robot, get_time_since_reset):
        self._robot = robot
        self.get_time_since_reset = get_time_since_reset

    @abstractmethod
    def update_controller_params(self, params):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def setup_ui_params(self, pybullet_client):
        pass

    @abstractmethod
    def read_ui_params(self, pybullet_client, ui):
        pass

    @abstractmethod
    def reset(self):
        pass
