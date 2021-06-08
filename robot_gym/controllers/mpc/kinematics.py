import numpy as np


class Kinematics:

    def __init__(self,
                 robot
                 ):

        self._pybullet_client = robot.pybullet_client
        self._robot = robot

    def _compute_jacobian(self, link_id):
        """Computes the Jacobian matrix for the given link.
        Args:
          link_id: The link id as returned from loadURDF.
        Returns:
          The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
          robot. For a _quadruped, the first 6 columns of the matrix corresponds to
          the CoM translation and rotation. The columns corresponds to a leg can be
          extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
        """
        all_joint_angles = [state[0] for state in self._robot.GetJointStates]
        # add arm joints, if any
        all_joint_angles.extend([0] * len(self._robot.arm_joints))
        zero_vec = [0] * len(all_joint_angles)
        jv, _ = self._pybullet_client.calculateJacobian(self._robot.GetRobotId, link_id,
                                                        (0, 0, 0), all_joint_angles,
                                                        zero_vec, zero_vec)
        jacobian = np.array(jv)
        assert jacobian.shape[0] == 3
        return jacobian

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        assert len(self._robot.GetFootLinkIds) == self._robot.GetConstants().NUM_LEG
        return self._compute_jacobian(
            link_id=self._robot.GetFootLinkIds[leg_id]
        )

    def MapContactForceToJointTorques(self, leg_id, contact_force):
        """Maps the foot contact force to the leg joint torques."""
        jv = self.ComputeJacobian(leg_id)
        all_motor_torques = np.matmul(contact_force, jv)
        motor_torques = {}
        motors_per_leg = self._robot.GetMotorConstants().NUM_MOTORS // self._robot.GetConstants().NUM_LEG
        com_dof = 6
        for joint_id in range(leg_id * motors_per_leg,
                              (leg_id + 1) * motors_per_leg):
            motor_torques[joint_id] = all_motor_torques[
                                          com_dof + joint_id
                                      ] * self._robot.GetMotorConstants().MOTOR_DIRECTION[joint_id]

        return motor_torques

    def joint_angles_from_link_position(
            self,
            link_position,
            link_id,
            joint_ids,
            position_in_world_frame,
            base_translation=(0, 0, 0),
            base_rotation=(0, 0, 0, 1)):
        """Uses Inverse Kinematics to calculate joint angles.
        Args:
          link_position: The (x, y, z) of the link in the body or the world frame,
            depending on whether the argument position_in_world_frame is true.
          link_id: The link id as returned from loadURDF.
          joint_ids: The positional index of the joints. This can be different from
            the joint unique ids.
          position_in_world_frame: Whether the input link_position is specified
            in the world frame or the robot's base frame.
          base_translation: Additional base translation.
          base_rotation: Additional base rotation.
        Returns:
          A list of joint angles.
        """
        if not position_in_world_frame:
            # Projects to local frame.
            base_position, base_orientation = self._pybullet_client.getBasePositionAndOrientation(
                self._robot.GetRobotId)  # robot.GetBasePosition(), robot.GetBaseOrientation()
            base_position, base_orientation = self._pybullet_client.multiplyTransforms(
                base_position, base_orientation, base_translation, base_rotation)

            # Projects to world space.
            world_link_pos, _ = self._pybullet_client.multiplyTransforms(
                base_position, base_orientation, link_position, self._robot.GetConstants().IDENTITY_ORIENTATION)
        else:
            world_link_pos = link_position

        ik_solver = 0
        all_joint_angles = self._pybullet_client.calculateInverseKinematics(
            self._robot.GetRobotId, link_id, world_link_pos, solver=ik_solver)

        # Extract the relevant joint angles.
        joint_angles = [all_joint_angles[i] for i in joint_ids]
        return joint_angles

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.
        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.
        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        return self._EndEffectorIK(
            leg_id, foot_local_position, position_in_world_frame=False)

    def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
        """Calculate the joint positions from the end effector position."""
        assert len(self._robot.GetFootLinkIds) == self._robot.GetConstants().NUM_LEG
        toe_id = self._robot.GetFootLinkIds[leg_id]
        motors_per_leg = self._robot.num_motors // self._robot.GetConstants().NUM_LEG
        joint_position_idxs = [
            i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg +
                             motors_per_leg)
        ]
        joint_angles = self.joint_angles_from_link_position(
            link_position=position,
            link_id=toe_id,
            joint_ids=joint_position_idxs,
            position_in_world_frame=position_in_world_frame)
        # Joint offset
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._robot.GetMotorConstants().MOTOR_OFFSET)[joint_position_idxs],
            self._robot.GetMotorConstants().MOTOR_DIRECTION[joint_position_idxs])
        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()
