import numpy as np

from robot_gym.model.equipment import camera
from robot_gym.util import pybullet_data


class Robot:

    def __init__(self,
                 pybullet_client,
                 mark,
                 simulation,
                 motor_control_mode,
                 z_offset=0.
                 ):

        self._pybullet_client = pybullet_client
        self._simulation = simulation
        self._z_offset = z_offset
        self._mark = mark
        self._marks = self.GetMarks()
        self._constants = self.GetConstants()
        self._num_motors = self._marks.MARK_PARAMS[self._mark]['num_motors']
        self._num_legs = self._marks.MARK_PARAMS[self._mark]['num_legs']
        self._motors_name = self._marks.MARK_PARAMS[self._mark]['motor_names']
        self._motor_enabled_list = self.GetMotorConstants().MOTOR_ENABLED
        self._motor_offset = self.GetMotorConstants().MOTOR_OFFSET
        self._motor_direction = self.GetMotorConstants().MOTOR_DIRECTION
        # load robot urdf
        self._quadruped = self._load_urdf()
        # build joints dict
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._BuildMotorIdList()
        # set robot init pose
        self.ResetPose()
        # fetch joints' states
        self.ReceiveObservation()
        # build locomotion motor model
        self._motor_model = self.GetMotorClass()(
            kp=self.GetMotorConstants().MOTOR_POSITION_GAINS,
            kd=self.GetMotorConstants().MOTOR_VELOCITY_GAINS,
            motor_control_mode=motor_control_mode,
            num_motors=self._num_motors
        )
        # robot equipment
        self._load_equipment()

    @property
    def num_legs(self):
        return self._num_legs

    @property
    def num_motors(self):
        return self._num_motors

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def equipment(self):
        return self._equip

    def set_up_discrete_action_space(self):
        pass

    def set_up_continuous_action_space(self):
        pass

    def GetBasePosition(self):
        """Get the position of Rex's base.
        Returns:
          The position of Rex's base.
        """
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self._quadruped))
        return position

    def GetBaseRollPitchYaw(self):
        """Get the orientation of Rex's base.
        Returns:
          The orientation of Rex's base.
        """
        _, orient = (self._pybullet_client.getBasePositionAndOrientation(self._quadruped))
        orient = self._pybullet_client.getEulerFromQuaternion(orient)
        return orient

    def GetMotorPositionGains(self):
        return self.GetMotorConstants().MOTOR_POSITION_GAINS

    def GetMotorVelocityGains(self):
        return self.GetMotorConstants().MOTOR_VELOCITY_GAINS

    def MapContactForceToJointTorques(self, leg_id, force):
        return self._simulation.controller.kinematics_model.MapContactForceToJointTorques(leg_id=leg_id,
                                                                                          contact_force=force)

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_position):
        return self._simulation.controller.kinematics_model.ComputeMotorAnglesFromFootLocalPosition(
            leg_id=leg_id,
            foot_local_position=foot_position
        )

    @property
    def GetJointStates(self):
        return self._joint_states

    @property
    def GetRobotId(self):
        return self._quadruped

    @property
    def GetFootLinkIds(self):
        return self._foot_link_ids

    @property
    def GetMotorModel(self):
        return self._motor_model

    def ReceiveObservation(self):
        self._joint_states = self._pybullet_client.getJointStates(self._quadruped, self._motor_id_list)

    def _load_urdf(self):
        x, y, z = self._constants.START_POS
        start_position = [x, y, z + self._z_offset]
        return self._pybullet_client.loadURDF(
            f"{pybullet_data.getDataPath()}/{self._marks.MARK_PARAMS[self._mark]['urdf_name']}", start_position)

    def ResetPose(self):
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self._quadruped,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for name, i in zip(self._motors_name, range(len(self._motors_name))):
            if self._constants.HIP_JOINTS_PATTERN in name:
                angle = self._constants.INIT_MOTOR_ANGLES[i] + self._constants.HIP_JOINT_OFFSET
            elif self._constants.UPPER_JOINTS_PATTERN in name:
                angle = self._constants.INIT_MOTOR_ANGLES[i] + self._constants.UPPER_LEG_JOINT_OFFSET
            elif self._constants.LOWER_JOINTS_PATTERN in name:
                angle = self._constants.INIT_MOTOR_ANGLES[i] + self._constants.LOWER_LEG_JOINT_OFFSET
            else:
                raise ValueError("The name %s is not recognized as a motor joint." %
                                 name)
            self._pybullet_client.resetJointState(
                self._quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

    def _GetMotorNames(self):
        return self._motors_name

    def _BuildMotorIdList(self):
        self._motor_id_list = [
            self._joint_name_to_id[motor_name]
            for motor_name in self._GetMotorNames()
        ]

    # def GetBaseRollPitchYaw(self):
    #     """Get minitaur's base orientation in euler angle in the world frame.
    #     Returns:
    #       A tuple (roll, pitch, yaw) of the base in world frame.
    #     """
    #     orientation = self.GetTrueBaseOrientation()
    #     roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    #     return np.asarray(roll_pitch_yaw)

    def GetHipPositionsInBaseFrame(self):
        return self._constants.DEFAULT_HIP_POSITIONS

    def GetBaseVelocity(self):
        """Get the linear velocity of minitaur's base.
        Returns:
          The velocity of minitaur's base.
        """
        velocity, _ = self._pybullet_client.getBaseVelocity(self._quadruped)
        return velocity

    def GetTrueBaseOrientation(self):
        pos, orn = self._pybullet_client.getBasePositionAndOrientation(
            self._quadruped)
        return orn

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """Transform the angular velocity from world frame to robot's frame.
        Args:
          angular_velocity: Angular velocity of the robot in world frame.
          orientation: Orientation of the robot represented as a quaternion.
        Returns:
          angular velocity of based on the given orientation.
        """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                        orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def GetBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the minitaur's base in euler angle.
        Returns:
          rate of (roll, pitch, yaw) change of the minitaur's base.
        """
        angular_velocity = self._pybullet_client.getBaseVelocity(self._quadruped)[1]
        orientation = self.GetTrueBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angular_velocity,
                                                         orientation)

    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self._quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[self._constants.BODY_B_FIELD_NUMBER] == self._quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(
                    contact[self._constants.LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts

    def GetMotorAngles(self):
        motor_angles = [state[0] for state in self._joint_states]
        motor_angles = np.multiply(
            np.asarray(motor_angles) - np.asarray(self._motor_offset),
            self._motor_direction)
        return motor_angles

    def GetTrueMotorAngles(self):
        """Gets the twelve motor angles at the current moment, mapped to [-pi, pi].
        Returns:
          Motor angles, mapped to [-pi, pi].
        """
        self.ReceiveObservation()

        return self.GetMotorAngles()

    def GetPDObservation(self):
        self.ReceiveObservation()
        observation = []
        observation.extend(self.GetTrueMotorAngles())
        observation.extend(self.GetTrueMotorVelocities())
        q = observation[0:self._num_motors]
        qdot = observation[self._num_motors:2 * self._num_motors]
        return np.array(q), np.array(qdot)

    def GetTrueMotorVelocities(self):
        """Get the velocity of all eight motors.
        Returns:
          Velocities of all eight motors.
        """
        motor_velocities = [state[1] for state in self._joint_states]

        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetTrueObservation(self):
        self.ReceiveObservation()
        observation = []
        observation.extend(self.GetTrueMotorAngles())
        observation.extend(self.GetTrueMotorVelocities())
        observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetTrueBaseOrientation())
        observation.extend(self.GetTrueBaseRollPitchYawRate())
        return observation

    def ApplyAction(self, motor_commands, motor_control_mode):
        """Apply the motor commands using the motor model.
        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands
          motor_control_mode: A MotorControlMode enum.
        """
        motor_commands = np.asarray(motor_commands)
        q, qdot = self.GetPDObservation()
        qdot_true = self.GetTrueMotorVelocities()
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot, qdot_true, motor_control_mode)

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque,
                                                 self._motor_direction)
        motor_ids = []
        motor_torques = []

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                         self._applied_motor_torque,
                                                         self._motor_enabled_list):
            if motor_enabled:
                motor_ids.append(motor_id)
                motor_torques.append(motor_torque)
            else:
                motor_ids.append(motor_id)
                motor_torques.append(0)
        self._SetMotorTorqueByIds(motor_ids, motor_torques)

    def _SetMotorTorqueByIds(self, motor_ids, torques):
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self._quadruped,
            jointIndices=motor_ids,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            forces=torques)

    def _BuildJointNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self._quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self._quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.
        Raises:
          ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self._quadruped)
        self._chassis_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._knee_link_ids = []
        self._foot_link_ids = []

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self._quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if self._constants.CHASSIS_NAME_PATTERN.match(joint_name):
                self._chassis_link_ids.append(joint_id)
            elif self._constants.HIP_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            elif self._constants.UPPER_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            # We either treat the lower leg or the toe as the foot link, depending on
            # the urdf version used.
            elif self._constants.LOWER_NAME_PATTERN.match(joint_name):
                self._knee_link_ids.append(joint_id)
            elif self._constants.TOE_NAME_PATTERN.match(joint_name):
                # assert self._urdf_filename == URDF_WITH_TOES
                self._foot_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

        self._leg_link_ids.extend(self._knee_link_ids)
        self._leg_link_ids.extend(self._foot_link_ids)

        # assert len(self._foot_link_ids) == NUM_LEGS
        self._chassis_link_ids.sort()
        self._motor_link_ids.sort()
        self._knee_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()

        return

    def link_position_in_base_frame(self, link_id):
        """Computes the link's local position in the robot frame.
        Args:
          link_id: The link to calculate its relative position.
        Returns:
          The relative position of the link.
        """
        base_position, base_orientation = self._pybullet_client.getBasePositionAndOrientation(self._quadruped)
        inverse_translation, inverse_rotation = self._pybullet_client.invertTransform(
            base_position, base_orientation)

        link_state = self._pybullet_client.getLinkState(self._quadruped, link_id)
        link_position = link_state[0]
        link_local_position, _ = self._pybullet_client.multiplyTransforms(
            inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

        return np.array(link_local_position)

    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self._foot_link_ids

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        assert len(self._foot_link_ids) == self._num_legs
        foot_positions = []
        for foot_id in self.GetFootLinkIDs():
            foot_positions.append(
                self.link_position_in_base_frame(link_id=foot_id)
            )
        return np.array(foot_positions)

    def Terminate(self):
        pass

    def _load_equipment(self):
        self._equip = {}
        if "hardware" in self._marks.MARK_PARAMS[self._mark]:
            if "camera" in self._marks.MARK_PARAMS[self._mark]["hardware"]:
                self._equip = camera.parse_cams(self._marks, self._mark, self._equip)

    def update_equipment(self):
        if "cams" in self._equip:
            self._equip["cams"][self._equip["default_cam"]].get_camera_image(self._pybullet_client)

    def get_default_camera(self):
        return self._equip["cams"][self._equip["default_cam"]]
