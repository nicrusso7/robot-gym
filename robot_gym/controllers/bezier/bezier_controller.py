import time
import numpy as np
from robot_gym.controllers.pose import kinematics, constants
from robot_gym.controllers.bezier import constants as const

from robot_gym.controllers.controller import Controller


class BezierController(Controller):

    def __init__(self, robot, get_time_since_reset):
        super(BezierController, self).__init__(robot, get_time_since_reset)
        self._frame = np.zeros([4, 3])
        self._phi = 0.
        self._phi_stance = 0.
        self._last_time = 0.
        self._alpha = 0.
        self._s = False
        self.y_dist = 0.155
        self.x_dist = 0.23
        self.height = 0.22
        self._start_frames = np.asmatrix([[self.x_dist / 2, -self.y_dist / 2, -self.height],
                                          [self.x_dist / 2, self.y_dist / 2, -self.height],
                                          [-self.x_dist / 2, -self.y_dist / 2, -self.height],
                                          [-self.x_dist / 2, self.y_dist / 2, -self.height]])
        # if mode == "walk":
        #     self._offset = np.array([0., 0.5, 0.5, 0.])
        #     self.step_offset = 0.5
        # else:
        #     self._offset = np.array([0., 0., 0.8, 0.8])
        #     self.step_offset = 0.5

        # self._offset = np.array([0.5, 0., 0., 0.5])
        # self.step_offset = 0.5

        self._offset = np.array([0., 0., 0.8, 0.8])
        self.step_offset = 0.5
        # controller inputs
        self._step_length = .0
        self._step_angle = .0
        self._step_rotation = .0
        self._step_period = .0

    @classmethod
    def get_constants(cls):
        del cls
        return const

    @staticmethod
    def solve_bin_factor(n, k):
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

    def bezier_curve(self, t, k, point):
        n = 11
        return point * self.solve_bin_factor(n, k) * np.power(t, k) * np.power(1 - t, n - k)

    @staticmethod
    def calculate_stance(phi_st, v, angle):
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        A = 0.001
        half_l = 0.05
        p_stance = half_l * (1 - 2 * phi_st)
        stance_x = c * p_stance * np.abs(v)
        stance_y = -s * p_stance * np.abs(v)
        stance_z = -A * np.cos(np.pi / (2 * half_l) * p_stance)
        return stance_x, stance_y, stance_z

    def calculate_bezier_swing(self, phi_sw, v, angle, direction):
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        X = np.abs(v) * c * np.array([-0.04, -0.056, -0.06, -0.06, -0.06, 0.,
                                      0., 0., 0.06, 0.06, 0.056, 0.04]) * direction
        Y = np.abs(v) * s * (-X)
        Z = np.abs(v) * np.array([0., 0., 0.0405, 0.0405, 0.0405, 0.0405,
                                  0.0405, 0.0495, 0.0495, 0.0495, 0., 0.])

        # X = np.abs(v) * c * np.array([-0.07,
        #                               -0.08,
        #                               -0.09,
        #                               -0.09,
        #                               0.,
        #                               0.,
        #                               0.09,
        #                               0.09,
        #                               0.08,
        #                               0.07])
        #
        # Y = np.abs(v) * s * np.array([0.0,
        #                               0.0,
        #                               0.0,
        #                               0.0,
        #                               0.,
        #                               -0.,
        #                               -0.0,
        #                               -0.0,
        #                               -0.0,
        #                               -0.0])
        #
        # Z = np.abs(v) * np.array([0.,
        #                           0.,
        #                           0.07,
        #                           0.07,
        #                           0.07,
        #                           0.08,
        #                           0.08,
        #                           0.08,
        #                           0.,
        #                           0.])
        swing_x = 0.
        swing_y = 0.
        swing_z = 0.

        for i in range(12):
            swing_x = swing_x + self.bezier_curve(phi_sw, i, X[i])
            swing_y = swing_y + self.bezier_curve(phi_sw, i, Y[i])
            swing_z = swing_z + self.bezier_curve(phi_sw, i, Z[i])
        return swing_x, swing_y, swing_z

    def step_trajectory(self, phi, v, angle, w_rot, center_to_foot, direction):
        if phi >= 1:
            phi = phi - 1.
        r = np.sqrt(center_to_foot[0] ** 2 + center_to_foot[1] ** 2)
        foot_angle = np.arctan2(center_to_foot[1], center_to_foot[0])
        if w_rot >= 0.:
            circle_trajectory = 90. - np.rad2deg(foot_angle - self._alpha)
        else:
            circle_trajectory = 270. - np.rad2deg(foot_angle - self._alpha)

        if phi <= self.step_offset:
            # stance phase
            phi_stance = phi / self.step_offset
            stepX_long, stepY_long, stepZ_long = self.calculate_stance(phi_stance, v, angle)
            stepX_rot, stepY_rot, stepZ_rot = self.calculate_stance(phi_stance, w_rot, circle_trajectory)
        else:
            # swing phase
            phiSwing = (phi - self.step_offset) / (1 - self.step_offset)
            stepX_long, stepY_long, stepZ_long = self.calculate_bezier_swing(phiSwing, v, angle, direction)
            stepX_rot, stepY_rot, stepZ_rot = self.calculate_bezier_swing(phiSwing, w_rot, circle_trajectory, direction)
        if center_to_foot[1] > 0:
            if stepX_rot < 0:
                self._alpha = -np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
            else:
                self._alpha = np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
        else:
            if stepX_rot < 0:
                self._alpha = np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
            else:
                self._alpha = -np.arctan2(np.sqrt(stepX_rot ** 2 + stepY_rot ** 2), r)
        coord = np.empty(3)
        coord[0] = stepX_long + stepX_rot
        coord[1] = stepY_long + stepY_rot
        coord[2] = stepZ_long + stepZ_rot
        return coord

    def loop(self, v, angle, w_rot, t, direction, frames=None):
        if frames is None:
            frames = self._start_frames
        if t <= 0.01:
            t = 0.01
        if self._phi >= 0.99:
            self._last_time = time.time()
        self._phi = (time.time() - self._last_time) / t
        step_coord = self.step_trajectory(self._phi + self._offset[0], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[0, :])), direction)  # FR
        self._frame[0, 0] = frames[0, 0] + step_coord[0]
        self._frame[0, 1] = frames[0, 1] + step_coord[1]
        self._frame[0, 2] = frames[0, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[1], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[1, :])), direction)  # FL
        self._frame[1, 0] = frames[1, 0] + step_coord[0]
        self._frame[1, 1] = frames[1, 1] + step_coord[1]
        self._frame[1, 2] = frames[1, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[2], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[2, :])), direction)  # RR
        self._frame[2, 0] = frames[2, 0] + step_coord[0]
        self._frame[2, 1] = frames[2, 1] + step_coord[1]
        self._frame[2, 2] = frames[2, 2] + step_coord[2]

        step_coord = self.step_trajectory(self._phi + self._offset[3], v, angle, w_rot,
                                          np.squeeze(np.asarray(frames[3, :])), direction)  # RL
        self._frame[3, 0] = frames[3, 0] + step_coord[0]
        self._frame[3, 1] = frames[3, 1] + step_coord[1]
        self._frame[3, 2] = frames[3, 2] + step_coord[2]
        return self._frame

    def update_controller_params(self, params):
        step_length, step_angle, step_rotation, step_period = params
        self.loop(step_length, step_angle, step_rotation, step_period, 1.0)

    def get_action(self):
        orientation = [.0, .0, .0]
        position = [.0, .0, .0]
        foot_front_right = np.asarray([self._frame[0, 0], self._frame[0, 1], self._frame[0, 2]])
        foot_front_left = np.asarray([self._frame[1, 0], self._frame[1, 1], self._frame[1, 2]])
        foot_rear_right = np.asarray([self._frame[2, 0], self._frame[2, 1], self._frame[2, 2]])
        foot_rear_left = np.asarray([self._frame[3, 0], self._frame[3, 1], self._frame[3, 2]])
        # rotation vertices
        hip_front_right_vertex = kinematics.transform(constants.hip_front_right_v, orientation, position)
        hip_front_left_vertex = kinematics.transform(constants.hip_front_left_v, orientation, position)
        hip_rear_right_vertex = kinematics.transform(constants.hip_rear_right_v, orientation, position)
        hip_rear_left_vertex = kinematics.transform(constants.hip_rear_left_v, orientation, position)
        # leg vectors
        front_right_coord = foot_front_right - hip_front_right_vertex
        front_left_coord = foot_front_left - hip_front_left_vertex
        rear_right_coord = foot_rear_right - hip_rear_right_vertex
        rear_left_coord = foot_rear_left - hip_rear_left_vertex
        # leg vectors transformation
        inv_orientation = orientation
        inv_position = position
        t_front_right_coord = kinematics.transform(front_right_coord, inv_orientation, inv_position)
        t_front_left_coord = kinematics.transform(front_left_coord, inv_orientation, inv_position)
        t_rear_right_coord = kinematics.transform(rear_right_coord, inv_orientation, inv_position)
        t_rear_left_coord = kinematics.transform(rear_left_coord, inv_orientation, inv_position)
        # solve IK
        front_right_angles = kinematics.solve_IK(t_front_right_coord, constants.hip, constants.leg, constants.foot, True)
        front_left_angles = kinematics.solve_IK(t_front_left_coord, constants.hip, constants.leg, constants.foot, False)
        rear_right_angles = kinematics.solve_IK(t_rear_right_coord, constants.hip, constants.leg, constants.foot, True)
        rear_left_angles = kinematics.solve_IK(t_rear_left_coord, constants.hip, constants.leg, constants.foot, False)

        signal = [
            front_right_angles[0], front_right_angles[1], front_right_angles[2],
            front_left_angles[0], front_left_angles[1], front_left_angles[2],
            rear_right_angles[0], rear_right_angles[1], rear_right_angles[2],
            rear_left_angles[0], rear_left_angles[1], rear_left_angles[2]
        ]
        return signal

    def setup_ui_params(self, pybullet_client):
        step_length = pybullet_client.addUserDebugParameter("step_length", -1.5, 1.5, 0.)
        step_rotation = pybullet_client.addUserDebugParameter("step_rotation", -1.5, 1.5, 0.)
        step_angle = pybullet_client.addUserDebugParameter("step_angle", -180., 180., 0.)
        step_period = pybullet_client.addUserDebugParameter("step_period", -1., 1., 0.)
        return step_length, step_rotation, step_angle, step_period

    def read_ui_params(self, pybullet_client, ui):
        step_length, step_rotation, step_angle, step_period = ui
        step_length = pybullet_client.readUserDebugParameter(step_length)
        step_rotation = pybullet_client.readUserDebugParameter(step_rotation)
        step_angle = pybullet_client.readUserDebugParameter(step_angle)
        step_period = pybullet_client.readUserDebugParameter(step_period)
        return step_length, step_rotation, step_angle, step_period

    def reset(self):
        pass
