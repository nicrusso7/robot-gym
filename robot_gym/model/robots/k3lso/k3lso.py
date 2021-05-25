from robot_gym.model.robots.k3lso import constants, marks, motor_constants, ctrl_constants
from robot_gym.model.robots.robot import Robot


class K3lso(Robot):

    @classmethod
    def GetMotorClass(cls):
        del cls
        return motor_constants.MOTOR_CONTROL_CLASS

    @classmethod
    def GetMotorConstants(cls):
        del cls
        return motor_constants

    @classmethod
    def GetCtrlConstants(cls):
        del cls
        return ctrl_constants

    @classmethod
    def GetConstants(cls):
        del cls
        return constants

    @classmethod
    def GetMarks(cls):
        del cls
        return marks
