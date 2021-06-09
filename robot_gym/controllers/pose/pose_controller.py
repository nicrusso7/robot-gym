import numpy as np

from robot_gym.controllers.arm import arm_controller
from robot_gym.controllers.controller import Controller
from robot_gym.controllers.pose import kinematics
from robot_gym.model.robots import simple_motor


class PoseController(Controller):

    MOTOR_CONTROL_MODE = simple_motor.MOTOR_CONTROL_POSITION

    def __init__(self, robot, get_time_since_reset):
        super(PoseController, self).__init__(robot, get_time_since_reset)
        self._arm_ctrl = arm_controller.ArmController(robot, get_time_since_reset)
        self._constants = robot.GetCtrlConstants()
        self._orientation = [0, 0, 0]
        self._position = [0, 0, 0]
        self._frames = np.asmatrix([[self._constants.x_dist / 2, -self._constants.y_dist / 2, -self._constants.height],
                                    [self._constants.x_dist / 2, self._constants.y_dist / 2, -self._constants.height],
                                    [-self._constants.x_dist / 2, -self._constants.y_dist / 2, -self._constants.height],
                                    [-self._constants.x_dist / 2, self._constants.y_dist / 2, -self._constants.height]])

    def update_controller_params(self, params):
        self._position, self._orientation, arm_target = params
        self._arm_ctrl.update_controller_params(arm_target)

    def setup_ui_params(self, pybullet_client):
        base_x = pybullet_client.addUserDebugParameter("base_x", -.02, .02, 0.)
        base_y = pybullet_client.addUserDebugParameter("base_y", -.02, .02, 0.)
        base_z = pybullet_client.addUserDebugParameter("base_z", -.065, .03, 0.)
        roll = pybullet_client.addUserDebugParameter("roll", -np.pi / 4, np.pi / 4, 0)
        pitch = pybullet_client.addUserDebugParameter("pitch", -np.pi / 4, np.pi / 4, 0)
        yaw = pybullet_client.addUserDebugParameter("yaw", -np.pi / 4, np.pi / 4, 0)
        arm_ui = self._arm_ctrl.setup_ui_params(pybullet_client)
        return base_x, base_y, base_z, roll, pitch, yaw, arm_ui

    def read_ui_params(self, pybullet_client, ui):
        base_x, base_y, base_z, roll, pitch, yaw, arm_params = ui
        position = np.array(
            [
                pybullet_client.readUserDebugParameter(base_x),
                pybullet_client.readUserDebugParameter(base_y),
                pybullet_client.readUserDebugParameter(base_z)
            ]
        )
        orientation = np.array(
            [
                pybullet_client.readUserDebugParameter(roll),
                pybullet_client.readUserDebugParameter(pitch),
                pybullet_client.readUserDebugParameter(yaw)
            ]
        )
        arm_params = self._arm_ctrl.read_ui_params(pybullet_client, arm_params)
        return position, orientation, arm_params

    def reset(self):
        pass

    def get_action(self):
        foot_front_right = np.asarray([self._frames[0, 0], self._frames[0, 1], self._frames[0, 2]])
        foot_front_left = np.asarray([self._frames[1, 0], self._frames[1, 1], self._frames[1, 2]])
        foot_rear_right = np.asarray([self._frames[2, 0], self._frames[2, 1], self._frames[2, 2]])
        foot_rear_left = np.asarray([self._frames[3, 0], self._frames[3, 1], self._frames[3, 2]])
        # rotation vertices
        hip_front_right_vertex = kinematics.transform(self._constants.hip_front_right_v, self._orientation, self._position)
        hip_front_left_vertex = kinematics.transform(self._constants.hip_front_left_v, self._orientation, self._position)
        hip_rear_right_vertex = kinematics.transform(self._constants.hip_rear_right_v, self._orientation, self._position)
        hip_rear_left_vertex = kinematics.transform(self._constants.hip_rear_left_v, self._orientation, self._position)
        # leg vectors
        front_right_coord = foot_front_right - hip_front_right_vertex
        front_left_coord = foot_front_left - hip_front_left_vertex
        rear_right_coord = foot_rear_right - hip_rear_right_vertex
        rear_left_coord = foot_rear_left - hip_rear_left_vertex
        # leg vectors transformation
        inv_orientation = -self._orientation
        inv_position = -self._position
        t_front_right_coord = kinematics.transform(front_right_coord, inv_orientation, inv_position)
        t_front_left_coord = kinematics.transform(front_left_coord, inv_orientation, inv_position)
        t_rear_right_coord = kinematics.transform(rear_right_coord, inv_orientation, inv_position)
        t_rear_left_coord = kinematics.transform(rear_left_coord, inv_orientation, inv_position)
        # solve IK
        front_right_angles = kinematics.solve_IK(t_front_right_coord, self._constants.hip, self._constants.leg, self._constants.foot, True)
        front_left_angles = kinematics.solve_IK(t_front_left_coord, self._constants.hip, self._constants.leg, self._constants.foot, False)
        rear_right_angles = kinematics.solve_IK(t_rear_right_coord, self._constants.hip, self._constants.leg, self._constants.foot, True)
        rear_left_angles = kinematics.solve_IK(t_rear_left_coord, self._constants.hip, self._constants.leg, self._constants.foot, False)

        # t_front_right = hip_front_right_vertex + t_front_right_coord
        # t_front_left = hip_front_left_vertex + t_front_left_coord
        # t_rear_right = hip_rear_right_vertex + t_rear_right_coord
        # t_rear_left = hip_rear_left_vertex + t_rear_left_coord
        # t_frames = np.asmatrix([[t_front_right[0], t_front_right[1], t_front_right[2]],
        #                         [t_front_left[0], t_front_left[1], t_front_left[2]],
        #                         [t_rear_right[0], t_rear_right[1], t_rear_right[2]],
        #                         [t_rear_left[0], t_rear_left[1], t_rear_left[2]]])
        signal = [
            front_right_angles[0], front_right_angles[1], front_right_angles[2],
            front_left_angles[0], front_left_angles[1], front_left_angles[2],
            rear_right_angles[0], rear_right_angles[1], rear_right_angles[2],
            rear_left_angles[0], rear_left_angles[1], rear_left_angles[2]
        ]

        arm_signal = self._arm_ctrl.get_action()
        signal.extend(arm_signal[12:])

        return signal
