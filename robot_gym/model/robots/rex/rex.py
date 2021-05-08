from robot_gym.model.robots.rex import constants, marks, motor_constants
from robot_gym.model.robots.robot import Robot


class Rex(Robot):

    @classmethod
    def GetMotorClass(cls):
        del cls
        return motor_constants.MOTOR_CONTROL_CLASS

    @classmethod
    def GetMotorConstants(cls):
        del cls
        return motor_constants

    @classmethod
    def GetConstants(cls):
        del cls
        return constants

    @classmethod
    def GetMarks(cls):
        del cls
        return marks
