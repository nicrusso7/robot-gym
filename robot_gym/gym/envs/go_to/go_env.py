""" This Gym Environment implements the "Go to target" task. """
import math
import random
import time

from gym import spaces
import numpy as np
from robot_gym.core import sim_constants

from robot_gym.gym import robot_gym_env
from robot_gym.gym.envs.go_to.path_follower.follower import Follower
from robot_gym.gym.envs.go_to.path_follower.line_interpolation import sort_points, interpolate_points
from robot_gym.gym.envs.go_to.path_follower.path import Path
from robot_gym.gym.envs.go_to.path_planner import potential_field_planner
from robot_gym.util import pybullet_data


class GoEnv(robot_gym_env.RobotGymEnv):
    """
    "points_latch" - returns flattened array shape (2 * num_cam_pts,) if at least 2 line points
        are visible in camera window, otherwise returns previous observation

    The gym environment for the rex.

    It simulates the locomotion of a rex, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the rex walks in 2000 steps and penalizes the energy
    expenditure or how near rex is to the target position.

    """

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 target_position=None,
                 obstacles_list=None,
                 terrain_id=None,
                 terrain_type='plane',
                 show_plot=False,
                 on_rack=False,
                 render=False,
                 record_video=False,
                 debug=False,
                 policy=False):

        """
        Initialize the Walk To gym environment.

        Args:
          on_rack: Whether to place the rex on rack. This is only used to debug
            the walk gait. In this mode, the rex's base is hung midair so
            that its walk gait is clearer to visualize.
          render: Whether to render the simulation.
        """

        super(GoEnv, self).__init__(
            robot_model=robot_model,
            on_rack=on_rack,
            render=render,
            debug=debug,
            terrain_id=terrain_id,
            terrain_type=terrain_type,
            record_video=record_video,
            controller_class=controller_class,
            mark=mark,
            policy=policy
        )

        self._debug = debug
        self._show_plot = show_plot
        self._target_position = target_position
        self.simulation.pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
        self._obstacles_list = np.array(obstacles_list) if obstacles_list is not None else np.array([])
        self._random_target = True if self._target_position is None else False

        # number of line representation points
        self._num_cam_pts = 8
        # maximum episode time in seconds
        self._max_time = 90
        # observation type
        self._obs_type = "points_latch"

        self._done = False
        self._plot = None
        self._path = None
        self._follower = None
        self._observation = []

        # build action space
        self._build_action_space()

        # build observation space
        self._build_observation_space()

        if self._debug:
            self._ui = self._setup_ui_parameters()
            self._build_world(True)

    def _build_action_space(self):
        self.action_space = spaces.Box(low=np.array([-0.4]),
                                       high=np.array([0.4]), dtype=np.float32)

    def _build_observation_space(self):
        self.observation_space = spaces.Box(low=np.array([0.0, -0.2] * self._num_cam_pts),
                                            high=np.array([0.3, 0.2] * self._num_cam_pts),
                                            dtype=np.float32)

    def _setup_ui_parameters(self):
        ui = {
            # controller ui params
            "ctrl_ui": self.simulation.controller.setup_ui_params(self.simulation.pybullet_client),
            # sim ui params
            "sim_ui": self.simulation.setup_ui_parameters(),
            # # terrain ui params
            # "terrain": self.simulation.terrain.setup_ui_params(self.simulation.pybullet_client),
            # robot camera
            "cam_ui": self.setup_equipment_ui_params()
        }
        return ui

    def setup_equipment_ui_params(self):
        cam_ui = self.simulation.pybullet_client.addUserDebugParameter("Show robot cam", 0, -1, 0)
        return cam_ui

    def parse_equipment_ui_params(self):
        if "cams" in self.simulation.robot.equipment:
            cam_flag = self.simulation.pybullet_client.readUserDebugParameter(self._ui["cam_ui"])
            if cam_flag % 2 != 0:
                return True
            return False
        return False

    def _read_inputs(self):
        return self.simulation.controller.read_ui_params(
            self.simulation.pybullet_client,
            self._ui["ctrl_ui"]
        )

    def reset(self):
        # create path
        self._build_path()

        if self._debug:
            # draw path line
            self._draw_path_line()
            # create world objects
            self._build_world()
            # setup sim camera position
            self.simulation.set_camera(3.8, 0, -30)

        self._done = False

        # reset plot (if any)
        if self._plot:
            plt.close(self._plot["fig"])
            self._plot = None

        return super(GoEnv, self).reset(orientation=self._path.start_angle)

    def _build_path(self):
        if self._random_target:
            # randomize target
            x = round(random.uniform(-2.5, 2.5), 2)
            y = round(random.uniform(-2.5, 2.5), 2)
            if 1. > x > 0:
                x = 1.
            if -1. < x < 0:
                x = -1.
            if 1. > y > 0:
                y = 1.
            if -1. < y < 0:
                y = -1.
            self._target_position = [x, y]
            if self._debug:
                print(f"Target: {x},{y}")
        # build path points lists
        self._calculate_path_points()
        # create path object
        self._path = Path(pts=self._path_points, debug=self._debug)
        # create follower object
        self._follower = Follower(self._num_cam_pts,
                                  self._path.start_xy,
                                  self.simulation.robot.GetBaseRollPitchYaw()[2])

    def _draw_path_line(self):
        if "target_line" in self.world_object:
            # erase old path
            self.simulation.pybullet_client.removeAllUserDebugItems()
        start_x, start_y = self._path_points[0]
        for p in self._path_points:
            x, y = p
            self.world_object["target_line"] = self.simulation.pybullet_client.addUserDebugLine(
                (start_x, start_y, 0.), (x, y, 0.), (1., 0., 0.), 10)
            start_x, start_y = x, y

    def _calculate_path_points(self):
        target_x, target_y = self._target_position
        robot_x, robot_y, _ = self.simulation.robot.GetBasePosition()
        if len(self._obstacles_list) > 0:
            obstacle_x, obstacle_y = self._obstacles_list[:, 0], self._obstacles_list[:, 1]
        else:
            obstacle_x = []
            obstacle_y = []
        x, y = potential_field_planner.get_path(target_x, target_y, obstacle_x, obstacle_y)
        self._path_points = np.stack((x, y), axis=-1)

    def _distance_to_target(self):
        obs = self.get_observation()
        distance = math.sqrt(((obs[3] - self._target_position[0]) ** 2 +
                              (obs[4] - self._target_position[1]) ** 2))
        return distance

    def reward(self):
        return self._follower.reward(self._path)

    def close(self):
        if self._plot:
            plt.close(self._plot["fig"])
            plt.ioff()
        return

    def termination(self):
        if super(GoEnv, self).termination()[0]:
            return True, {}
        done = False
        track_err = self._path.distance_from_point(self._follower.pos[0])
        max_steps = self._max_time / (sim_constants.SIMULATION_TIME_STEP * sim_constants.ACTION_REPEAT)
        if self._path.done:
            done = True
            print("FULL PATH DONE!")
        elif self._on_target():
            done = True
            print("PATH DONE!")
        elif abs(self._follower._position_on_track - self._path.progress) > 0.5:
            done = True
            print("PROGRESS DISTANCE LIMIT")
        elif track_err > self._follower.max_track_err:
            done = True
            print("TRACK DISTANCE LIMIT")
        elif self.simulation.step_counter > max_steps:
            done = True
            print("TIME LIMIT")

        info = self._follower.get_info()
        return done, info

    def get_observation(self):
        self._follower.update_position_velocity(self.simulation.robot.GetBasePosition(),
                                                self.simulation.robot.GetBaseRollPitchYaw(),
                                                self.simulation.robot._quadruped,
                                                self.simulation.pybullet_client)

        visible_pts = self._follower.cam_window.visible_points(self._path.mpt)
        if len(visible_pts) > 0:
            # TODO too slow! optimize interpolation
            visible_pts_local = self._follower.cam_window.convert_points_to_local(visible_pts)
            visible_pts_local = sort_points(visible_pts_local)
            visible_pts_local = interpolate_points(visible_pts_local, self._follower.num_cam_pts)
            if len(visible_pts_local) > 1:
                observation = visible_pts_local.flatten().tolist()
                self._observation = observation
                return observation
            else:
                # return previous observation
                return self._observation
        else:
            # return previous observation
            return self._observation

    def step(self, action, **kwargs):
        if self._debug and not self.simulation.read_ui_parameters(self._ui) and not self._policy:
            # overwrite agent action with UI input
            action = self._read_inputs()
            # update cameras position (if any)

        if self.parse_equipment_ui_params():
            # update follower cam
            pos_x, pos_y = self._follower.cam_pos_point.get_xy()
            target_x, target_y = self._follower.cam_target_point.get_xy()
            self.simulation.robot.get_default_camera().position = pos_x, pos_y, 0.095
            self.simulation.robot.get_default_camera().target = target_x, target_y, 0.0
            kwargs = {"update_equip": True}

        if not isinstance(action, tuple):
            action = (0.3, action)

        # print(action)
        if self._on_target():
            action = self.simulation.controller.get_standing_action()

        if self._show_plot:
            self._update_plot()
        return super(GoEnv, self).step(action, **kwargs)

    def _on_target(self):
        if self._distance_to_target() <= 0.15:
            return True
        return False

    def _build_world(self, start_pos=None):
        if "target" not in self.world_object:
            urdf_root = pybullet_data.getDataPath()
            self.world_object["target"] = self.simulation.pybullet_client.loadURDF(
                f"{urdf_root}/world/objects/target/target.urdf"
            )
        if start_pos is None:
            x, y = self._target_position
            target_pos = [x, y, 0.]
        else:
            target_pos = [0., 0., 0.]
        self.simulation.pybullet_client.resetBasePositionAndOrientation(self.world_object["target"],
                                                                        target_pos,
                                                                        [0, 0, 0, 1])

    def _update_plot(self):
        if self._plot is None:
            global plt
            import matplotlib.pyplot as plt
            plt.ion()

            fig, (track_ax, win_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
            track_ax.axis("off")
            # win_ax.axis("off")
            track_ax.set_aspect("equal")
            win_ax.set_aspect("equal")
            track_ax.set_xlim(-1.5, 1.5)
            track_ax.set_ylim(-1, 1)
            track_ax.plot(self._path.x, self._path.y, "k--")
            win_ax.plot(*self._follower.cam_window.get_local_window().plottable, "m-")

            pos_line, = track_ax.plot(0, 0, "ro")
            win_line, = track_ax.plot(*self._follower.cam_window.plottable, "m--")
            track_ref_line, = track_ax.plot(*self._follower.track_ref_point.plottable, "c.")
            vis_pts_line, = win_ax.plot([], [], "c.")
            progress_line, = track_ax.plot([], [], "g--")

            self._plot = {"fig": fig,
                          "track_ax": track_ax,
                          "win_ax": win_ax,
                          "pos_line": pos_line,
                          "win_line": win_line,
                          "track_ref_line": track_ref_line,
                          "vis_pts_line": vis_pts_line,
                          "progress_line": progress_line}

        # Plot data
        (x, y), yaw = self._follower.format_position(self.simulation.robot.GetBasePosition(),
                                                     self.simulation.robot.GetBaseRollPitchYaw())
        self._plot["pos_line"].set_xdata(x)
        self._plot["pos_line"].set_ydata(y)

        self._plot["win_line"].set_xdata(self._follower.cam_window.plottable[0])
        self._plot["win_line"].set_ydata(self._follower.cam_window.plottable[1])

        vis_pts = np.array(self._observation).reshape((-1, 2))

        self._plot["vis_pts_line"].set_xdata(vis_pts[:, 0])
        self._plot["vis_pts_line"].set_ydata(vis_pts[:, 1])

        self._plot["track_ref_line"].set_xdata(self._follower.track_ref_point.plottable[0])
        self._plot["track_ref_line"].set_ydata(self._follower.track_ref_point.plottable[1])

        self._plot["progress_line"].set_xdata(self._path.pts[0:self._path.progress_idx, 0])
        self._plot["progress_line"].set_ydata(self._path.pts[0:self._path.progress_idx, 1])

        plt.draw()
        plt.pause(0.0001)
