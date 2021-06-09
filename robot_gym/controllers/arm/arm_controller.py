from gibson2.external.pybullet_tools.utils import get_max_limits, get_min_limits, get_joint_positions

from robot_gym.controllers.controller import Controller

import numpy as np


class ArmController(Controller):

    def __init__(self, robot, get_time_since_reset):
        super(ArmController, self).__init__(robot, get_time_since_reset)
        self._end_effector_target = robot.get_end_effector_position()
        self._rest = False

    def setup_ui_params(self, pybullet_client):
        end_eff_x = pybullet_client.addUserDebugParameter("arm_target_x", -1.0, 1.0, self._end_effector_target[0])
        end_eff_y = pybullet_client.addUserDebugParameter("arm_target_y", -1.0, 1.0, self._end_effector_target[1])
        end_eff_z = pybullet_client.addUserDebugParameter("arm_target_z", -1.0, 1.0, self._end_effector_target[2])
        rest_pose = pybullet_client.addUserDebugParameter("Arm rest pose", 0, -1, 0)
        return end_eff_x, end_eff_y, end_eff_z, rest_pose

    def read_ui_params(self, pybullet_client, ui):
        end_eff_x, end_eff_y, end_eff_z, rest_pose = ui
        end_eff_target = [
            pybullet_client.readUserDebugParameter(end_eff_x),
            pybullet_client.readUserDebugParameter(end_eff_y),
            pybullet_client.readUserDebugParameter(end_eff_z),
        ]
        rest = self._parse_rest_pose_param(rest_pose, pybullet_client)
        return end_eff_target, rest

    @staticmethod
    def _parse_rest_pose_param(ui, pybullet_client):
        rest_flag = pybullet_client.readUserDebugParameter(ui)
        if rest_flag % 2 != 0:
            return True
        return False

    def update_controller_params(self, params):
        self._end_effector_target, self._rest = params

    def get_action(self):
        if self._rest:
            return self._robot.GetConstants().INIT_MOTOR_ANGLES
        return self._calculate_accurate_ik(self._robot.pybullet_client, self._robot.GetRobotId, self._robot.arm_joints,
                                           self._robot.arm_gripper,
                                           self._end_effector_target)

    @staticmethod
    def _calculate_accurate_ik(pybullet_client, robot_id, arm_joints,
                               end_effector_id, target_pos, threshold=0.01, max_iter=100):
        it = 0
        max_limits = get_max_limits(robot_id, arm_joints)
        min_limits = get_min_limits(robot_id, arm_joints)
        rest_position = list(get_joint_positions(robot_id, arm_joints))
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        # jd = [0.1 for _ in joint_range]
        while it < max_iter:
            joint_poses = pybullet_client.calculateInverseKinematics(
                robot_id,
                end_effector_id,
                target_pos,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position
                )

            ls = pybullet_client.getLinkState(robot_id, end_effector_id)
            newPos = ls[4]

            dist = np.linalg.norm(np.array(target_pos) - np.array(newPos))
            if dist < threshold:
                break

            it += 1
        return joint_poses

    def reset(self):
        pass
