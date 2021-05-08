""" Robot agnostic gym env """
import time

import gym
import numpy as np
from abc import ABC, abstractmethod

from gym.utils import seeding

from robot_gym.core.simulation import Simulation


class RobotGymEnv(gym.Env, ABC):
    """ A robot (and Task) agnostic gym environment. """

    metadata = {"render.modes": ["human", "gui", "rgb_array", "pov"], "video.frames_per_second": 100}

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 terrain_id=None,
                 terrain_type='plane',
                 on_rack=False,
                 render=False,
                 record_video=False,
                 debug=False,
                 policy=False
                 ):
        """
        Initialize the gym environment.

        Args:
          on_rack: Whether to place Spotmicro on rack. This is only used to debug
            the walk gait. In this mode, Spotmicro's base is hanged midair so
            that its walk gait is clearer to visualize.
          render: Whether to render the simulation.
        """
        self._render = render
        self._debug = debug
        self._policy = policy
        self._last_frame_time = 0.
        # start simulation
        self.world_object = {}
        self._simulation = Simulation(robot_model=robot_model,
                                      debug=debug,
                                      terrain_id=terrain_id,
                                      terrain_type=terrain_type,
                                      record_video=record_video,
                                      controller_class=controller_class,
                                      render=render,
                                      mark=mark)
        # Set up logging
        # @TODO implement logging
        self._on_rack = on_rack
        self.seed()

    @property
    def simulation(self):
        return self._simulation

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def _build_action_space(self):
        pass

    @abstractmethod
    def _build_observation_space(self):
        pass

    def close(self):
        self._simulation.robot.Terminate()

    def reset(self, **kwargs):
        print("reset simulation")
        self.simulation.reset()

        if self.simulation.terrain.terrain_type is not "plane":
            self.simulation.terrain.update_terrain()

        if "position" in kwargs:
            position = kwargs["position"]
        else:
            position = self._simulation.robot.GetConstants().START_POS

        # get terrain offset
        z_offset = self.simulation.terrain.get_terrain_z_offset()
        position = [position[0], position[1], position[2] + z_offset]

        if "orientation" in kwargs:
            orientation = [0, 0, kwargs["orientation"]]
        else:
            orientation = self._simulation.robot.GetConstants().INIT_ORIENTATION

        # @TODO call logger

        self._simulation.pybullet_client.resetBasePositionAndOrientation(
            self._simulation.robot.GetRobotId,
            position,
            self._simulation.pybullet_client.getQuaternionFromEuler(orientation)
        )
        self._simulation.robot.ResetPose()
        self._simulation.SettleRobotDownForReset(reset_time=1.0)
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, **kwargs):
        """Step forward the simulation, given the action."""
        # self._sleep_at_reset()
        self.simulation.controller.update_controller_params(action)
        action = self.simulation.controller.get_action()
        self._simulation.ApplyStepAction(action)
        if "update_equip" in kwargs:
            self._simulation.robot.update_equipment()
        observation = self.get_observation()
        reward = self.reward()
        done, info = self.termination()
        # @TODO call logger
        return np.array(observation), reward, done, info

    def render(self, mode="rgb_array", close=False):
        return self._simulation.Render(mode)

    def _sleep_at_reset(self):
        if self._render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.simulation.env_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.simulation.robot.GetBasePosition()

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self.simulation.pybullet_client.getDebugVisualizerCamera()[8:11]
            self.simulation.pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            self.simulation.pybullet_client.configureDebugVisualizer(
                self.simulation.pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    def all_sensors(self):
        """Returns all robot and environmental sensors."""
        return self.simulation.robot.sensors

    def is_falling(self):
        """Decide whether the robot is falling.

        If all the foot contacts are False then terminate the episode.

        Returns:
          Boolean value that indicates whether the robot is falling.
        """
        contacts = self.simulation.robot.GetFootContacts()
        return not any(contacts)

    def termination(self):
        if self.is_falling():
            print("FALLING DOWN!")
        return self.is_falling(), {}
