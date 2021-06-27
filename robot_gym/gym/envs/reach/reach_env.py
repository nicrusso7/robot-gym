import gym
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.render.profiler import Profiler

import numpy as np

from gibson2.scenes.empty_scene import EmptyScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.stadium_scene import StadiumScene

from robot_gym.controllers.mpc import mpc_controller
from robot_gym.core import sim_constants

from robot_gym.controllers.pose import pose_controller

from robot_gym.model.robots.ghost import ghost

import pybullet as p
from robot_gym.core.simulation import Simulation


class ReachEnv(iGibsonEnv):

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 config,
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
        self._config = config
        self._debug = debug
        self._run_policy = policy
        super().__init__(self._config, mode=mode, action_timestep=sim_constants.SIMULATION_TIME_STEP,
                         physics_timestep=sim_constants.SIMULATION_TIME_STEP)
        if self._debug:
            self._ui = self._setup_ui_parameters()

    def _setup_ui_parameters(self):
        ui = {
            # controller ui params
            "ctrl_ui": self._base_sim.controller.setup_ui_params(self._base_sim.pybullet_client),
            # sim ui params
            "sim_ui": self._base_sim.setup_ui_parameters(),
        }
        return ui

    def step(self, action):
        if self._debug and not self._base_sim.read_ui_parameters(self._ui) and not self._run_policy:
            # overwrite agent action with UI input
            action = self._read_inputs()
        return super().step(action)

    def _read_inputs(self):
        return self._base_sim.controller.read_ui_params(
            self._base_sim.pybullet_client,
            self._ui["ctrl_ui"]
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

        self._base_sim.robot = self._robot_model(pybullet_client=p,
                                                 mark=self._mark,
                                                 simulation=self._base_sim,
                                                 motor_control_mode=self._controller_class.MOTOR_CONTROL_MODE,
                                                 load_model=False)

        # TODO edit here to support multiple robots (e.g. multiple agents)
        self.robots = [self._base_sim._robot]
        for robot in self.robots:
            self.simulator.import_robot(robot)
        # TODO -----------------------------------------------------------

        # setup locomotion controller
        self._base_sim.controller = self._controller_class(self._base_sim.robot,
                                                           self._base_sim.GetTimeSinceReset)

        self._base_sim.reset()

        action_high = self._base_sim.controller.ACTION_HIGH
        # keep arm in rest position
        action_dim = self._base_sim.controller.ACTION_DIM - 3
        self._base_sim.robot.action_high = np.array([action_high] * action_dim)
        self._base_sim.robot.action_low = -self._base_sim.robot.action_high
        self._base_sim.robot.action_space = gym.spaces.Box(shape=(action_dim,),
                                                           low=-action_high,
                                                           high=action_high,
                                                           dtype=np.float32)

        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def land(self, obj, pos, orn):
        super().land(obj, pos, orn)
        self._base_sim.robot.ResetPose()

    # def run_simulation(self):
    #     """
    #     Run simulation for one action timestep (same as one render timestep in Simulator class)
    #
    #     :return: collision_links: collisions from last physics timestep
    #     """
    #     for _ in range(0):
    #         p.stepSimulation()
    #     # self.simulator.sync()
    #     collision_links = list(p.getContactPoints(
    #         bodyA=self.robots[0].robot_ids[0]))
    #     return self.filter_collision_links(collision_links)


def main():
    env = ReachEnv(ghost.Ghost, "1", mpc_controller.MPCController, "reach.yaml", debug=True, render=True)
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
