import time
import pybullet
from pybullet_utils import bullet_client
import pybullet_data as pd
import numpy as np
import random
import gym
from mpc_controller import state_estimator as st_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import openloop_gait_generator
from mpc_controller import offset_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller import locomotion_controller

from robots import a1_robot as robot_sim
from worlds import plane_world, slope_world, stair_world, uneven_world

WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                               slope=slope_world.SlopeWorld,
                               stair=stair_world.StairWorld,
                               uneven=uneven_world.UnevenWorld)

def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
  vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
  wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER

  time_points = (0, 5, 10, 15, 20, 25,30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                  (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(
      time_points,
      speed_points,
      kind="previous",
      fill_value="extrapolate",
      axis=0)(
          t)

  return speed[0:3], speed[3]

class RL_Env(gym.Env):
    def __init__(self, show_gui=False, world=None):
        self.EPISODE_LEN_SEC = 12
        self.CTRL_FREQ = 50
        self.action_high = np.array([4, 0.99, 0.35, 0.25, 0.1, 0.1, 0.1, 0.1])
        self.action_low = np.array([0.1, 0.01, 0.1, 0.01,-0.1,-0.1,-0.1,-0.1])
        self.desired_speed, self.desired_twisting_speed = (0.45, 0, 0), 0
        self.world = world
        self.show_gui = show_gui
        if show_gui:
            p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pd.getDataPath())
        self.simulation_time_step = 0.002
        self.SIM_FREQ = int(1/self.simulation_time_step)
        self.pybullet_client = p
        # construct robot class
        robot_uid = p.loadURDF("data/a1.urdf", robot_sim.START_POS)

        self._robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=self.simulation_time_step)
        self._clock = lambda: self._robot.GetTimeSinceReset()
        self._gait_generator = offset_gait_generator.OffsetGaitGenerator(
              self._robot, [0., np.pi, np.pi, 0.])
        self._state_estimator = st_estimator.COMVelocityEstimator(self._robot, velocity_window_size=60, ground_normal_window_size=10)
        self._swing_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            self._state_estimator,
            desired_speed=self.desired_speed,
            desired_twisting_speed=self.desired_twisting_speed,
            desired_height= robot_sim.MPC_BODY_HEIGHT,
            foot_height=0.17,
            foot_landing_clearance=-0.01)

        self._stance_controller = torque_stance_leg_controller.TorqueStanceLegController(
            self._robot,
            self._gait_generator,
            self._state_estimator,
            desired_speed=self.desired_speed,
            desired_twisting_speed=self.desired_twisting_speed,
            desired_body_height=robot_sim.MPC_BODY_HEIGHT,
            body_mass=robot_sim.MPC_BODY_MASS,
            body_inertia=robot_sim.MPC_BODY_INERTIA)

        # self.lin_speed, self.ang_speed = _generate_example_linear_angular_speed(self._robot.GetTimeSinceReset)

        self.reset()

    def reset(self):
        # reset pybullet
        p = self.pybullet_client
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=30)
        p.setTimeStep(self.simulation_time_step)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setAdditionalSearchPath(pd.getDataPath())
        terain = random.choice(["plane", "slope", "stair","uneven"])
        if self.world != None:
            terain = self.world
        world_class=WORLD_NAME_TO_CLASS_MAP[terain]
        world = world_class(p)
        world.build_world()
        robot_uid = p.loadURDF("data/a1.urdf", robot_sim.START_POS)
        self._robot.ResetPose(robot_uid)
        if self.show_gui:
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RENDERING, 1)

        self._reset_time = self._clock()
        self._time_since_reset = 0
        self._gait_generator.reset(self._time_since_reset)
        self._state_estimator.reset(self._time_since_reset)
        self._swing_controller.reset(self._time_since_reset)
        self._stance_controller.reset(self._time_since_reset)

        return self.get_observation() # 27

    def get_observation(self):

        base_height = self._robot.GetTrueBasePosition()[2:]
        robot_orientation = self._robot.GetBaseRollPitchYaw()
        robot_velocity = np.array(self._robot.GetBaseVelocity())
        # if not self.use_real_robot:
        #   robot_velocity *= np.random.uniform(0.8, 1.2)
        robot_rpy_rate = self._robot.GetBaseRollPitchYawRate() # 3
        foot_position = self._robot.GetFootPositionsInBaseFrame().flatten()  # 12
        foot_contact_state = np.array(
            [(leg_state in (gait_generator_lib.LegState.STANCE,
                            gait_generator_lib.LegState.EARLY_CONTACT,
                            gait_generator_lib.LegState.LOSE_CONTACT))
             for leg_state in self._gait_generator.leg_state], dtype=np.int32)
        # lin_speed, ang_speed = _generate_example_linear_angular_speed(self._time_since_reset)

        return np.concatenate(
          (
              base_height,  # 1
              robot_orientation,  # 3
              robot_velocity,  # 3
              robot_rpy_rate,  # 3
              foot_position,  # 12
              foot_contact_state, #4
              self.desired_speed[:1],  # 1
          ),
          axis=-1)

    def step(self, action):
        """
        DEFINE ACTION:
        Gait config: f: frequesy , stance_duration
        desired speed: lin_speed, ang_speed

        """
        self._time_since_reset = self._clock() - self._reset_time
        # lin_speed,  ang_speed  = (0.45, 0, 0), 0
        print("original action", action)
        action_mid = (self.action_low + self.action_high) / 2
        action_range = (self.action_high - self.action_low) / 2
        action = (action * action_range + action_mid).squeeze()
        print("rescale action", action)
        f = action[0]
        sw_ratio = action[1]
        desired_height = action[2]
        foot_height = action[3]
        desired_speed_offset = action[4:7]
        desired_twisting_speed_offset = action[7]

        self._gait_generator.gait_params = [f, np.pi, np.pi, 0, sw_ratio]
        self._gait_generator.update(self._time_since_reset)

        self._state_estimator.update(self._time_since_reset)
        future_contacts = self._gait_generator.get_estimated_contact_states(
        self._stance_controller._PLANNING_HORIZON_STEPS, self._stance_controller._PLANNING_TIMESTEP)


        self._swing_controller.desired_speed = self.desired_speed + desired_speed_offset
        self._swing_controller.desired_twisting_speed = self.desired_twisting_speed + desired_twisting_speed_offset
        self._swing_controller.desired_height = desired_height
        self._swing_controller.foot_height = foot_height
        self._swing_controller.update(self._time_since_reset)

        self._stance_controller.desired_speed = self.desired_speed + desired_speed_offset
        self._stance_controller.desired_twisting_speed = self.desired_twisting_speed + desired_twisting_speed_offset
        self._stance_controller._desired_body_height = desired_height
        self._stance_controller.update(self._time_since_reset, future_contact_estimate=future_contacts)

         # Get robot action and step the robot
        self.robot_action, self.qp_sol = self.get_action()
        self._robot.Step(self.robot_action)
        if self.show_gui:
            self.pybullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=30 + self._robot.GetBaseRollPitchYaw()[2] / np.pi * 180,
                cameraPitch=-30,
                cameraTargetPosition=self._robot.GetTrueBasePosition(),
            )
        reward = self._reward_fn(action)
        done = not self.is_safe
        return self.get_observation(), reward, done, dict()

    def _reward_fn(self, action):
        # del action # unused
        desired_lin_speed, desired_ang_rate = np.array((0.45, 0., 0.)), np.array((0, 0, 0))

        actual_lin_speed = self._robot.GetBaseVelocity()
        tracking_lin_speed = np.linalg.norm(desired_lin_speed - actual_lin_speed)
        actual_ang_rate = self._robot.GetBaseRollPitchYawRate()
        tracking_ang_rate = np.linalg.norm(desired_ang_rate - actual_ang_rate)
        reward = 10 - 5*tracking_lin_speed # - tracking_ang_rate
        # reward = max(0, 1 - tracking_lin_speed) + max(0, 1 - tracking_ang_rate)
        if not self.is_safe:
            reward = -10

        return reward

    @property
    def is_safe(self):
        rot_mat = np.array(
            self._robot.pybullet_client.getMatrixFromQuaternion(
                self._state_estimator.com_orientation_quat_ground_frame)).reshape(
                    (3, 3))
        up_vec = rot_mat[2, 2]
        base_height = self._robot.GetTrueBasePosition()[2]

        return up_vec > 0.85 and base_height > 0.18

    def get_action(self):
        """Returns the control ouputs (e.g. positions/torques) for all motors."""
        swing_action = self._swing_controller.get_action()
        stance_action, qp_sol = self._stance_controller.get_action()

        action = []
        for joint_id in range(self._robot.num_motors):
            if joint_id in swing_action:
                action.extend(swing_action[joint_id])
            else:
                assert joint_id in stance_action
                action.extend(stance_action[joint_id])
        action = np.array(action, dtype=np.float32)

        return action, dict(qp_sol=qp_sol)
