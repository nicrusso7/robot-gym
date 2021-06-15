import time

from robot_gym.controllers.bezier import bezier_controller
from robot_gym.controllers.pose import pose_controller

from robot_gym.controllers.mpc import mpc_controller
from robot_gym.core.simulation import Simulation
from robot_gym.io.gamepad import xbox_one_pad
from robot_gym.model.robots.ghost import ghost


class Playground:

    def __init__(self, robot_model, mark, record_video=False, gamepad=False, pybullet_client=None):
        self._robot_model = robot_model
        self._mark = mark
        self._record_video = record_video
        self._gamepad = gamepad
        self._current_ctrl = pose_controller.PoseController
        if self._gamepad:
            gamepad = xbox_one_pad.XboxGamepad()
            self._command_function = gamepad.get_command
        # create simulation
        self._create_simulation(False, pybullet_client, robot_model, mark, self._current_ctrl, "plane", None)

    def _create_simulation(self, record_video, pybullet_client, robot_model,
                           mark, controller_class, terrain_type, terrain_id):
        # start simulation
        self._sim = Simulation(robot_model=robot_model,
                               debug=True,
                               terrain_id=terrain_id,
                               terrain_type=terrain_type,
                               record_video=record_video,
                               mark=mark,
                               controller_class=controller_class,
                               render=True,
                               pybullet_client=pybullet_client)
        # setup ui
        self._ui = self._setup_ui_parameters()

    def _setup_ui_parameters(self):
        ui = {
            # controller class ui params
            "ctrl_class": self.setup_ctrl_class_params(),
            # controller commands ui commands
            "ctrl_cmd": self._sim.controller.setup_ui_params(self._sim.pybullet_client),
            # terrain ui params
            "terrain": self._sim.terrain.setup_ui_params(self._sim.pybullet_client),
            # robot equipment
            "equip": self.setup_equipment_ui_params()
        }
        return ui

    def setup_ctrl_class_params(self):
        mcp_ctrl = self._sim.pybullet_client.addUserDebugParameter("MPC controller", 0, -1, 0)
        pose_ctrl = self._sim.pybullet_client.addUserDebugParameter("Pose controller", 0, -1, 0)
        bezier_ctrl = self._sim.pybullet_client.addUserDebugParameter("Bezier controller", 0, -1, 0)
        return mcp_ctrl, pose_ctrl, bezier_ctrl

    def parse_ctrl_class_params(self, ui):
        mpc_flag = self._sim.pybullet_client.readUserDebugParameter(ui[0])
        pose_flag = self._sim.pybullet_client.readUserDebugParameter(ui[1])
        bezier_flag = self._sim.pybullet_client.readUserDebugParameter(ui[2])
        if mpc_flag % 2 != 0:
            self._current_ctrl = mpc_controller.MPCController
            return True, self._current_ctrl
        elif pose_flag % 2 != 0:
            self._current_ctrl = pose_controller.PoseController
            return True, self._current_ctrl
        elif bezier_flag % 2 != 0:
            self._current_ctrl = bezier_controller.BezierController
            return True, self._current_ctrl
        return False, self._current_ctrl

    def _update_world(self):
        refresh_ctrl, ctrl_class = self.parse_ctrl_class_params(self._ui["ctrl_class"])
        refresh_terrain, terrain_id, terrain_type = self._sim.terrain.parse_ui_input(self._ui["terrain"],
                                                                                     self._sim.pybullet_client)
        args = {}
        if refresh_ctrl or refresh_terrain:
            args = {
                "controller_class": ctrl_class,
                "terrain_id": terrain_id,
                "terrain_type": terrain_type
            }
        return refresh_ctrl or refresh_terrain, args

    def _parse_ctrl_input(self):
        if self._gamepad:
            # read gamepad input
            params = self._command_function()
        else:
            # read ui input
            params = self._sim.controller.read_ui_params(self._sim.pybullet_client, self._ui["ctrl_cmd"])
        # Updates the controller behavior parameters.
        self._sim.controller.update_controller_params(params)

    def run(self):
        start_time = self._sim.GetTimeSinceReset()
        current_time = start_time
        while self._sim.pybullet_client.isConnected():
            # time.sleep(0.0008) # on some fast computer, works better with sleep on real robot?
            start_time_robot = current_time
            start_time_wall = time.time()
            # update the sim
            restart, args = self._update_world()
            if restart:
                self._reset(args)
            if self.parse_equipment_ui_params(self._ui["equip"]):
                # show cam view
                self._sim.robot.update_equipment()
            # parse user input
            self._parse_ctrl_input()
            # get controller generated action
            action = self._sim.controller.get_action()
            # apply action to robot
            self._sim.ApplyStepAction(action)
            if self.is_falling() and \
                    self._current_ctrl.MOTOR_CONTROL_MODE is mpc_controller.MPCController.MOTOR_CONTROL_MODE:
                # robot is falling down, reset the simulation.
                self._reset(None)
            current_time = self._sim.GetTimeSinceReset()
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if actual_duration < expected_duration:
                time.sleep(expected_duration - actual_duration)
            # print("actual_duration=", actual_duration)

    def _reset(self, args):
        if args is None:
            args = {
                "controller_class": self._current_ctrl,
                "terrain_id": self._sim.terrain.terrain_id,
                "terrain_type": self._sim.terrain.terrain_type
            }
        self._sim.pybullet_client.resetSimulation()
        self._sim.pybullet_client.removeAllUserParameters()
        self._create_simulation(False, self._sim.pybullet_client, self._robot_model, self._mark, **args)

    def setup_equipment_ui_params(self):
        cam_ui = self._sim.pybullet_client.addUserDebugParameter("Show robot cam", 0, -1, 0)
        return cam_ui

    def parse_equipment_ui_params(self, ui):
        cam_flag = self._sim.pybullet_client.readUserDebugParameter(ui)
        if cam_flag % 2 != 0:
            return True
        return False

    def is_falling(self):
        contacts = self._sim.robot.GetFootContacts()
        return not any(contacts)


if __name__ == "__main__":
    playground = Playground(ghost.Ghost, "1", False)
    playground.run()
