import time
import pybullet
from pybullet_utils import bullet_client
import pybullet_data as pd
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

class RL_Env(gym.Env):
    def __init__(self, show_gui=False):
        self.obs =
        self.action =
        self.show_gui = show_gui
        if show_gui:
            p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pd.getDataPath())
        self.simulation_time_step = 0.001
        self.pybullet_client = p
        # construct robot class
        robot_uid = p.loadURDF("data/a1.urdf", robot_sim.START_POS)

        self._robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=self.simulation_time_step)
        self._clock = lambda: self._robot.GetTimeSinceReset
        self._gait_generator = offset_gait_generator.OffsetGaitGenerator(
              self._robot, [0., np.pi, np.pi, 0.])
        self._state_estimator = st_estimator.COMVelocityEstimator(robot, velocity_window_size=60, ground_normal_window_size=10)
        self._swing_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_height= robot_sim.MPC_BODY_HEIGHT,
            foot_height=0.17,
            foot_landing_clearance=-0.01)

        self._stance_controller = torque_stance_leg_controller.TorqueStanceLegController(
            robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_body_height=robot_sim.MPC_BODY_HEIGHT,
            body_mass=robot_sim.MPC_BODY_MASS,
            body_inertia=robot_sim.MPC_BODY_INERTIA)

        self.reset()

    def reset(self):
        # reset pybullet
        p = self.pybullet_client
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=30)
        p.setTimeStep(0.001)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setAdditionalSearchPath(pd.getDataPath())
        terain = random.choice(["plane", "slope", "stair"])
        world_class=WORLD_NAME_TO_CLASS_MAP[terain]
        world = world_class(p)
        world.build_world()

        self._robot.reset()
        if self.show_gui:
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_RENDERING, 1)

        self._reset_time = self._clock()
        self._time_since_reset = 0
        self._gait_generator.reset(self._time_since_reset)
        self._state_estimator.reset(self._time_since_reset)
        self._swing_controller.reset(self._time_since_reset)
        self._stance_controller.reset(self._time_since_reset)

        return self.get_observation()
    # def update_desired_speed(self, lin_speed, ang_speed):
    #     self._swing_controller.desired_speed = lin_speed
    #     self._swing_controller.desired_twisting_speed = ang_speed
    #     self._stance_controller.desired_speed = lin_speed
    #     self._stance_controller.desired_twisting_speed = ang_speed

    # def get_observation(self):
    #     gait_generator_state = self._gait_generator.get_observation()  # 16
    #     base_height = self.robot.base_position[2:]
    #     robot_orientation = self.robot.base_orientation_rpy
    #     robot_velocity = np.array(self.robot.base_velocity)
    #     if not self.use_real_robot:
    #       robot_velocity *= np.random.uniform(0.8, 1.2)
    #     robot_rpy_rate = self.robot.base_angular_velocity_body_frame  # 3
    #     foot_position = self.robot.foot_positions_in_base_frame.flatten()  # 12
    #     desired_velocity = self.get_desired_speed(self._time_since_reset)
    #
    #     if self.config.use_full_observation:
    #       return np.concatenate(
    #           (
    #               gait_generator_state,  # 16
    #               base_height,  # 1
    #               robot_orientation,  # 3
    #               robot_velocity,  # 3
    #               robot_rpy_rate,  # 3
    #               foot_position,  # 12
    #               desired_velocity[:1],  # 1
    #           ),
    #           axis=-1,
    #       )

    def step(self, action):
        """
        DEFINE ACTION:
        Gait config: f: frequesy , stance_duration
        desired speed: lin_speed, ang_speed

        """
        self._time_since_reset = self._clock() - self._reset_time
        f = action[0]
        sw_ratio = action[1] #meadn = 0.5 std =
        desired_height = action[2]
        foot_height = action[3]
        desired_speed_offset = action[4]
        desired_twisting_speed_offset = action[5]
        self._gait_generator.gait_params = [f, np.pi, np.pi, 0, sw_ratio]
        self._gait_generator.update(self._time_since_reset)

        self._state_estimator.update(self._time_since_reset)
        future_contacts = self._gait_generator.get_estimated_contact_states(
        self._stance_controller.PLANNING_HORIZON_STEPS, self._stance_controller.PLANNING_TIMESTEP)


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
        sum_reward += self._reward_fn(action)
        done = not self.is_safe
        if done:
            logging.info("Unsafe, terminating episode...")
            break
        return self.get_observation(), sum_reward, done, dict()
    def _reward_fn(self, )
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
        # start_time = time.time()
        stance_action, qp_sol = self._stance_controller.get_action()
        # print(time.time() - start_time)
        action = []
        for joint_id in range(self._robot.num_motors):
            if joint_id in swing_action:
                action.extend(swing_action[joint_id])
            else:
                assert joint_id in stance_action
          action.extend(stance_action[joint_id])
      action = np.array(action, dtype=np.float32)

      return action, dict(qp_sol=qp_sol)


    def _reward_fn(self, action):
        # del action # unused
        desired_speed = self.get_desired_speed(self._time_since_reset)[0]

        actual_speed = self.robot.base_velocity[0]

        motor_heat = 0.3 * self.robot.motor_torques**2
        motor_mech = self.robot.motor_torques * self.robot.motor_velocities

        # Maximize over 0 since the battery can not be charged by the motor yet
        power_penalty = np.maximum(motor_heat + motor_mech, 0)
        power_penalty = np.sum(power_penalty)
        if self.config.get('use_cot', False):
          power_penalty /= np.maximum(desired_speed, 0.3)

        action_norm_penalty = np.sum(np.maximum(np.abs(action[1:7]) - 0.5, 0))

        alive_bonus = self.config.get('alive_bonus', 3.)

        speed_penalty_type = self.config.get('speed_penalty_type',
                                             'symmetric_square')
        if speed_penalty_type == 'symmetric_square':
          speed_penalty = (desired_speed - actual_speed)**2
        elif speed_penalty_type == 'asymmetric_square':
          speed_penalty = np.maximum(desired_speed - actual_speed, 0)**2
        elif speed_penalty_type == 'soft_symmetric_square':
          speed_diff = np.abs(desired_speed - actual_speed)
          speed_penalty = np.maximum(speed_diff - 0.2, 0)**2
        else:
          speed_diff = np.abs(desired_speed - actual_speed) / np.maximum(
              actual_speed, 0.3)
          speed_diff = np.clip(speed_diff, -1, 1)
          speed_penalty = speed_diff**2

        # rew = alive_bonus - power_penalty * 0.0025 - np.maximum(
        #     (desired_speed - actual_speed), 0
        # )**2 - action_norm_penalty * self.config.get('action_penalty_weight', 0)
        rew = alive_bonus - \
            power_penalty * self.config.get('power_penalty_weight', 0.0025) - \
            speed_penalty * self.config.get('speed_penalty_weight', 1) - \
            action_norm_penalty * self.config.get('action_penalty_weight', 0)
    return rew
