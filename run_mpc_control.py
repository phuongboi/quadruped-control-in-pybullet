from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet
import random

from mpc_controller import state_estimator as st_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import openloop_gait_generator
from mpc_controller import offset_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller import locomotion_controller

from robots import a1_robot as robot_sim
from worlds import plane_world, slope_world, stair_world, uneven_world

FLAGS = flags.FLAGS

WORLD_NAME_TO_CLASS_MAP = dict(plane=plane_world.PlaneWorld,
                               slope=slope_world.SlopeWorld,
                               stair=stair_world.StairWorld,
                               uneven=uneven_world.UnevenWorld)

_NUM_SIMULATION_ITERATION_STEPS = 300


_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 50

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)
# temporary not use
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

def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  # gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
  #     robot,
  #     stance_duration=_STANCE_DURATION_SECONDS,
  #     duty_factor=_DUTY_FACTOR,
  #     initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
  #     initial_leg_state=_INIT_LEG_STATE)
  gait_generator = offset_gait_generator.OffsetGaitGenerator(
        robot, [0., np.pi, np.pi, 0.])
  state_estimator = st_estimator.COMVelocityEstimator(robot, velocity_window_size=60, ground_normal_window_size=10)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height= robot_sim.MPC_BODY_HEIGHT,
      foot_height=0.17,
      foot_landing_clearance=-0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller

def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed


def _run_example(max_time=_MAX_TIME_SECONDS):
  """Runs the locomotion controller example."""

  p = bullet_client.BulletClient(connection_mode=pybullet.GUI)


  p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  p.setAdditionalSearchPath(pd.getDataPath())
  p.setPhysicsEngineParameter(numSolverIterations=30) #same
  p.setPhysicsEngineParameter(enableConeFriction=1)
  simulation_time_step = 0.0012
  p.setTimeStep(simulation_time_step)

  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pd.getDataPath())

  world_class=WORLD_NAME_TO_CLASS_MAP["stair"]
  world = world_class(p)
  world.build_world()

  #robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)
  robot_uid = p.loadURDF("data/a1.urdf", robot_sim.START_POS)

  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step)

  controller = _setup_controller(robot)
  controller.reset()

  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

  current_time = robot.GetTimeSinceReset()

  while current_time < max_time:
    # pos,orn = p.getBasePositionAndOrientation(robot_uid)
    p.submitProfileTiming("loop")

    # Updates the controller behavior parameters.
    # lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time)
    lin_speed, ang_speed = (0.45, 0., 0.), 0.
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    hybrid_action, info = controller.get_action()

    robot.Step(hybrid_action)

    # if record_video:
    #   p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=30 + robot.GetBaseRollPitchYaw()[2] / np.pi * 180,
        cameraPitch=-30,
        cameraTargetPosition=robot.GetTrueBasePosition(),
    )
    #
    # time.sleep(0.001)
    print("debug")
    print("linear vel",robot.GetBaseVelocity())
    print("angle vel", robot.GetBaseRollPitchYawRate())
    current_time = robot.GetTimeSinceReset()
    p.submitProfileTiming()


def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  app.run(main)
