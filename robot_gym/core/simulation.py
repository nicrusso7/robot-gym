import pybullet
import pybullet_data as pd
import numpy as np
from pybullet_utils import bullet_client

from robot_gym.core import sim_constants

from robot_gym.model.world.terrain import Terrain


class Simulation:

    def __init__(self,
                 robot_model,
                 mark,
                 controller_class,
                 terrain_id=None,
                 terrain_type='plane',
                 record_video=False,
                 render=False,
                 debug=False,
                 pybullet_client=None,
                 sim_mode="classic"):

        self._sim_mode = sim_mode
        if self._sim_mode == "classic":
            self._init_classic_mode(robot_model, mark, controller_class, terrain_id, terrain_type,
                                    record_video, render, debug, pybullet_client)
        else:
            self._init_gibson_mode()


    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def controller(self):
        return self._controller_obj

    @property
    def terrain(self):
        return self._terrain

    @property
    def step_counter(self):
        return self._step_counter

    @step_counter.setter
    def step_counter(self, count):
        self._step_counter = count

    @property
    def env_time_step(self):
        return self._env_time_step

    def _init_classic_mode(self,
                           robot_model,
                           mark,
                           controller_class,
                           terrain_id=None,
                           terrain_type='plane',
                           record_video=False,
                           render=False,
                           debug=False,
                           pybullet_client=None, ):
        self._step_counter = 0
        self._state_action_counter = 0
        self._is_render = render
        self._robot_model = robot_model
        self._mark = mark
        self._env_time_step = sim_constants.ACTION_REPEAT * sim_constants.SIMULATION_TIME_STEP

        # create pybullet client
        self._pybullet_client = self._StartSimulation(record_video,
                                                      sim_constants.NUM_BULLET_SOLVER_ITERATIONS,
                                                      sim_constants.SIMULATION_TIME_STEP,
                                                      pybullet_client)

        self.build_world(controller_class, terrain_id, terrain_type)

        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, debug)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, debug)

    def _init_gibson_mode(self):
        pass

    def _StartSimulation(self, record_video, num_bullet_solver_iterations, simulation_time_step, pybullet_client=None):
        if pybullet_client is None:
            pybullet_client = self._init_pybullet_client(record_video)

        pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_RENDERING, 0)

        pybullet_client.setAdditionalSearchPath(pd.getDataPath())

        self.reset_simulation_config(pybullet_client, num_bullet_solver_iterations, simulation_time_step)

        return pybullet_client

    def build_world(self, controller_class, terrain_id, terrain_type):
        # generate terrain
        self._GenerateTerrain(terrain_id, terrain_type, self._pybullet_client)
        # get terrain offset
        z_offset = self._terrain.get_terrain_z_offset()
        # create robot
        self._robot = self._robot_model(pybullet_client=self._pybullet_client,
                                        mark=self._mark,
                                        simulation=self,
                                        motor_control_mode=controller_class.MOTOR_CONTROL_MODE,
                                        z_offset=z_offset)

        # setup locomotion controller
        self._controller_obj = controller_class(self._robot, self.GetTimeSinceReset)

        self.reset()
        # settle robot down
        self.SettleRobotDownForReset(reset_time=1.0)

    def reset(self):
        self._step_counter = 0
        self._state_action_counter = 0
        # setup robot controller
        self.controller.reset()

    def set_camera(self, roll, pitch, yaw):
        self.pybullet_client.resetDebugVisualizerCamera(
            roll,
            pitch,
            yaw,
            [0, 0, 0]
        )

    def _GenerateTerrain(self, terrain_id, terrain_type, _pybullet_client):
        self._terrain = Terrain(terrain_type, terrain_id)
        self._terrain.generate_terrain(_pybullet_client)

    def GetTimeSinceReset(self):
        return self._step_counter * sim_constants.SIMULATION_TIME_STEP

    def Render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=sim_constants.CAMERA_DISTANCE,
            yaw=sim_constants.CAMERA_YAW,
            pitch=sim_constants.CAMERA_PITCH,
            roll=0,
            upAxisIndex=2)
        aspect = float(self._robot.GetConstants().RENDER_WIDTH) / self._robot.GetConstants().RENDER_HEIGHT
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=aspect,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._robot.GetConstants().RENDER_WIDTH,
            height=self._robot.GetConstants().RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _StepInternal(self, action, motor_control_mode):
        self._robot.ApplyAction(action, motor_control_mode)
        self._pybullet_client.stepSimulation()
        self._state_action_counter += 1

    def ApplyStepAction(self, action):
        """ Steps simulation. """
        for i in range(sim_constants.ACTION_REPEAT):
            self._StepInternal(action, self.controller.MOTOR_CONTROL_MODE)
            self._step_counter += 1

    def SettleRobotDownForReset(self, reset_time):
        if reset_time <= 0:
            return
        for _ in range(500):
            self._StepInternal(
                self._robot.GetConstants().INIT_MOTOR_ANGLES,
                motor_control_mode=self._robot.GetMotorConstants().MOTOR_MODEL.MOTOR_CONTROL_POSITION)

    def setup_ui_parameters(self):
        agent_feedback = self.pybullet_client.addUserDebugParameter("Agent feedback", 0, -1, 0)
        return agent_feedback

    def read_ui_parameters(self, ui):
        agent_action = self.pybullet_client.readUserDebugParameter(ui["sim_ui"])
        if agent_action % 2 == 0:
            return False
        return True

    def _init_pybullet_client(self, record_video):
        pybullet_client = pybullet
        if record_video:
            # recording video requires ffmpeg in the path
            pybullet_client.connect(pybullet.GUI,
                                    options=f"--width={sim_constants.RENDER_WIDTH} "
                                            f"--height={sim_constants.RENDER_HEIGHT} "
                                            f"--mp4=\"test.mp4\" --mp4fps=100")
            pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        else:
            if self._is_render:
                pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                pybullet_client = bullet_client.BulletClient()
        return pybullet_client

    @staticmethod
    def reset_simulation_config(pybullet_client, num_bullet_solver_iterations, simulation_time_step):
        pybullet_client.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)

        pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)

        pybullet_client.setTimeStep(simulation_time_step)

        pybullet_client.setGravity(0, 0, -9.8)

        pybullet_client.resetDebugVisualizerCamera(sim_constants.CAMERA_DISTANCE,
                                                   sim_constants.CAMERA_YAW,
                                                   sim_constants.CAMERA_PITCH,
                                                   [0, 0, 0])