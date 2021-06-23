import gym
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging

import numpy as np

from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.stadium_scene import StadiumScene
from robot_gym.controllers.pose import pose_controller

from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.ghost import ghost

import pybullet as p
from robot_gym.core.simulation import Simulation
from robot_gym.gym.robot_gym_env import RobotGymEnv
from robot_gym.model.robots.robot import Robot


class ReachEnv(iGibsonEnv):

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 debug=False,
                 policy=False,
                 render=False,
                 ):

        if render:
            mode = 'gui'
        else:
            mode = 'headless'

        self._robot_model = robot_model
        self._mark = mark
        self._controller_class = controller_class
        self._debug = debug
        self._run_policy = policy
        super().__init__("reach.yaml", mode=mode
                         , action_timestep=0.1, physics_timestep=0.001
                         )

    def load(self):
        """
        Load the scene and robot
        """
        if self.config['scene'] == 'empty':
            scene = EmptyScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'stadium':
            scene = StadiumScene()
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'gibson':
            scene = StaticIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
            )
            self.simulator.import_scene(
                scene, load_texture=self.config.get('load_texture', True))
        elif self.config['scene'] == 'igibson':
            scene = InteractiveIndoorScene(
                self.config['scene_id'],
                waypoint_resolution=self.config.get(
                    'waypoint_resolution', 0.2),
                num_waypoints=self.config.get('num_waypoints', 10),
                build_graph=self.config.get('build_graph', False),
                trav_map_resolution=self.config.get(
                    'trav_map_resolution', 0.1),
                trav_map_erosion=self.config.get('trav_map_erosion', 2),
                trav_map_type=self.config.get('trav_map_type', 'with_obj'),
                pybullet_load_texture=self.config.get(
                    'pybullet_load_texture', False),
                texture_randomization=self.texture_randomization_freq is not None,
                object_randomization=self.object_randomization_freq is not None,
                object_randomization_idx=self.object_randomization_idx,
                should_open_all_doors=self.config.get(
                    'should_open_all_doors', False),
                load_object_categories=self.config.get(
                    'load_object_categories', None),
                load_room_types=self.config.get('load_room_types', None),
                load_room_instances=self.config.get(
                    'load_room_instances', None),
            )
            # TODO: Unify the function import_scene and take out of the if-else clauses
            first_n = self.config.get('_set_first_n_objects', -1)
            if first_n != -1:
                scene._set_first_n_objects(first_n)
            self.simulator.import_ig_scene(scene)

        self.scene = scene

        self._base_sim = Simulation(robot_model=self._robot_model,
                                    debug=self._debug,
                                    controller_class=self._controller_class,
                                    mark=self._mark,
                                    sim_mode="base",
                                    pybullet_client=p)

        # p.setPhysicsEngineParameter(numSolverIterations=30)
        # p.setPhysicsEngineParameter(enableConeFriction=0)

        self._base_sim._pybullet_client = p

        self._base_sim._robot = self._robot_model(pybullet_client=p,
                                                  mark=self._mark,
                                                  simulation=self._base_sim,
                                                  motor_control_mode=self._controller_class.MOTOR_CONTROL_MODE,
                                                  load_model=False)
        # don't need arm, remove it from action space (if any).
        # TODO read controller dimension and limit
        self._base_sim._robot.action_high = np.array([0.5] * 18)
        self._base_sim._robot.action_low = -self._base_sim._robot.action_high
        self._base_sim._robot.action_space = gym.spaces.Box(shape=(18,),
                                                            low=-0.5,
                                                            high=0.5,
                                                            dtype=np.float32)

        # TODO edit here to support multiple robots (e.g. multiple agents)
        self.robots = [self._base_sim._robot]
        for robot in self.robots:
            self._base_sim._robot._quadruped = self.simulator.import_robot(robot)[0]
        # TODO -----------------------------------------------------------

        # init robot
        self._base_sim.pybullet_client.resetBasePositionAndOrientation(
            self._base_sim.robot.GetRobotId,
            self._base_sim.robot.GetConstants().START_POS,
            self._base_sim.pybullet_client.getQuaternionFromEuler(self._base_sim.robot.GetConstants().INIT_ORIENTATION)
        )
        self._base_sim._robot.init_robot()

        # setup locomotion controller
        self._base_sim._controller_obj = self._controller_class(self._base_sim._robot, self._base_sim.GetTimeSinceReset)

        self._base_sim.reset()
        # settle robot down
        # TODO do we need this? gibson env already drop the robot in reset step.
        # self._base_sim.SettleRobotDownForReset(reset_time=1.0)

        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        for _ in range(10):
            p.stepSimulation()
        self.simulator.sync()
        collision_links = list(p.getContactPoints(
            bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    # def land(self, obj, pos, orn):
    #     """
    #     Land the robot or the object onto the floor, given a valid position and orientation
    #
    #     :param obj: an instance of robot or object
    #     :param pos: position
    #     :param orn: orientation
    #     """
    #     is_robot = isinstance(obj, BaseRobot)
    #
    #     self.set_pos_orn_with_z_offset(obj, pos, orn)
    #
    #     if is_robot:
    #         obj.robot_specific_reset()
    #         obj.keep_still()
    #
    #     body_id = obj.robot_ids[0] if is_robot else obj.body_id
    #
    #     land_success = False
    #     # land for maximum 1 second, should fall down ~5 meters
    #     max_simulator_step = int(1.0 / self.action_timestep)
    #     for _ in range(max_simulator_step):
    #         self.simulator_step()
    #         if len(p.getContactPoints(bodyA=body_id)) > 0:
    #             land_success = True
    #             break
    #
    #     if not land_success:
    #         print("WARNING: Failed to land")
    #
    #     if is_robot:
    #         obj.robot_specific_reset()
    #         obj.keep_still()



def main():
    env = ReachEnv(ghost.Ghost, "1", pose_controller.PoseController, debug=True, render=True)
    for j in range(1):
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
