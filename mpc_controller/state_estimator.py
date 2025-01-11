"""State estimator."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import numpy as np
from typing import Any, Sequence
import collections
import pybullet as p  # pytype: disable=import-error

_DEFAULT_WINDOW_SIZE = 20


class MovingWindowFilter(object):
  """A stable O(1) moving filter for incoming data streams.

  We implement the Neumaier's algorithm to calculate the moving window average,
  which is numerically stable.

  """
  def __init__(self, window_size: int, dim: int = 3):
    """Initializes the class.

    Args:
      window_size: The moving window size.
    """
    assert window_size > 0
    self._window_size = window_size
    self._value_deque = collections.deque(maxlen=window_size)
    # The moving window sum.
    self._sum = np.zeros(dim)
    # The correction term to compensate numerical precision loss during
    # calculation.
    self._correction = np.zeros(dim)

  def _neumaier_sum(self, value: np.ndarray):
    """Update the moving window sum using Neumaier's algorithm.

    For more details please refer to:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    Args:
      value: The new value to be added to the window.
    """

    new_sum = self._sum + value
    self._correction = np.where(
        np.abs(self._sum) >= np.abs(value),
        self._correction + (self._sum - new_sum) + value,
        self._correction + (value - new_sum) + self._sum)
    self._sum = new_sum

  def calculate_average(self, new_value: np.ndarray) -> np.ndarray:
    """Computes the moving window average in O(1) time.

    Args:
      new_value: The new value to enter the moving window.

    Returns:
      The average of the values in the window.

    """
    deque_len = len(self._value_deque)
    if deque_len < self._value_deque.maxlen:
      pass
    else:
      # The left most value to be subtracted from the moving sum.
      self._neumaier_sum(-self._value_deque[0])

    self._neumaier_sum(new_value)
    self._value_deque.append(new_value)

    return (self._sum + self._correction) / self._window_size


class COMVelocityEstimator(object):
  """Estimate the CoM velocity using on board sensors.


  Requires knowledge about the base velocity in world frame, which for example
  can be obtained from a MoCap system. This estimator will filter out the high
  frequency noises in the velocity so the results can be used with controllers
  reliably.

  """

  def __init__(
      self,
      robot: Any,
      velocity_window_size: int = _DEFAULT_WINDOW_SIZE,
      ground_normal_window_size: int = _DEFAULT_WINDOW_SIZE
  ):
    self._robot = robot
    self._velocity_window_size = velocity_window_size
    self._ground_normal_window_size = ground_normal_window_size
    self._ground_normal = np.array([0., 0., 1.])
    self.reset(0)

  def reset(self, current_time):
    del current_time
    # We use a moving window filter to reduce the noise in velocity estimation.
    self._velocity_filter = MovingWindowFilter(
        window_size=self._velocity_window_size)
    self._ground_normal_filter = MovingWindowFilter(
        window_size=self._ground_normal_window_size)

    self._com_velocity_world_frame = np.array((0, 0, 0))
    self._com_velocity_body_frame = np.array((0, 0, 0))

  def _compute_ground_normal(self, contact_foot_positions):
      """Computes the surface orientation in robot frame based on foot positions.
     Solves a least squares problem, see the following paper for details:
     https://ieeexplore.ieee.org/document/7354099
     """
      contact_foot_positions = np.array(contact_foot_positions)
      normal_vec = np.linalg.lstsq(contact_foot_positions, np.ones(4), rcond=None)[0]
      normal_vec /= np.linalg.norm(normal_vec)
      if normal_vec[2] < 0:
          normal_vec = -normal_vec
      return normal_vec

  def update(self, current_time):
    del current_time
    velocity = np.array(self._robot.GetBaseVelocity())

    self._com_velocity_world_frame = self._velocity_filter.calculate_average(velocity)

    base_orientation = self._robot.GetTrueBaseOrientation()
    _, inverse_rotation = self._robot.pybullet_client.invertTransform(
        (0, 0, 0), base_orientation)

    self._com_velocity_body_frame, _ = (
        self._robot.pybullet_client.multiplyTransforms(
            (0, 0, 0), inverse_rotation, self._com_velocity_world_frame,
            (0, 0, 0, 1)))
    # for slope env
    ground_normal_vector = self._compute_ground_normal(
    self._robot._foot_contact_history)
    self._ground_normal = self._ground_normal_filter.calculate_average(
        ground_normal_vector)
    self._ground_normal /= np.linalg.norm(self._ground_normal)

  @property
  def ground_normal(self):
    return self._ground_normal

  @property
  def gravity_projection_vector(self):
    _, world_orientation_ground_frame = p.invertTransform(
        [0., 0., 0.], self.ground_orientation_world_frame)
    return np.array(
        p.multiplyTransforms([0., 0., 0.], world_orientation_ground_frame,
                             [0., 0., 1.], [0., 0., 0., 1.])[0])

  @property
  def com_position_ground_frame(self):
    foot_contacts = self._robot.GetFootContacts().copy()
    if np.sum(foot_contacts) == 0:
      return np.array((0, 0, self._robot.MPC_BODY_HEIGHT))
    else:
      foot_positions_robot_frame = self._robot.GetFootPositionsInBaseFrame()
      ground_orientation_matrix_robot_frame = p.getMatrixFromQuaternion(
          self.ground_orientation_robot_frame)
      ground_orientation_matrix_robot_frame = np.array(
          ground_orientation_matrix_robot_frame).reshape((3, 3))
      foot_positions_ground_frame = (foot_positions_robot_frame.dot(
          ground_orientation_matrix_robot_frame.T))
      foot_heights = -foot_positions_ground_frame[:, 2]
      return np.array((
          0,
          0,
          np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts),
      ))

  @property
  def com_orientation_quat_ground_frame(self):
    _, orientation = p.invertTransform([0., 0., 0.],
                                       self.ground_orientation_robot_frame)
    return np.array(orientation)

  @property
  def com_velocity_ground_frame(self):
    _, world_orientation_ground_frame = p.invertTransform(
        [0., 0., 0.], self.ground_orientation_world_frame)
    return np.array(
        p.multiplyTransforms([0., 0., 0.], world_orientation_ground_frame,
                             self._com_velocity_world_frame,
                             [0., 0., 0., 1.])[0])

  @property
  def com_rpy_rate_ground_frame(self):
    com_quat_world_frame = p.getQuaternionFromEuler(self._robot.GetBaseRollPitchYawRate())
    _, world_orientation_ground_frame = p.invertTransform(
        [0., 0., 0.], self.ground_orientation_world_frame)
    _, com_quat_ground_frame = p.multiplyTransforms(
        [0., 0., 0.], world_orientation_ground_frame, [0., 0., 0.],
        com_quat_world_frame)
    return np.array(p.getEulerFromQuaternion(com_quat_ground_frame))

  @property
  def com_velocity_body_frame(self) -> Sequence[float]:
    """The base velocity projected in the body aligned inertial frame.

    The body aligned frame is a intertia frame that coincides with the body
    frame, but has a zero relative velocity/angular velocity to the world frame.

    Returns:
      The com velocity in body aligned frame.
    """
    return self._com_velocity_body_frame

  @property
  def com_velocity_world_frame(self) -> Sequence[float]:
    return self._com_velocity_world_frame

  @property
  def ground_orientation_robot_frame(self):
    normal_vec = self.ground_normal
    # print("normal vector", normal_vec)
    axis = np.array([-normal_vec[1], normal_vec[0], 0])
    axis /= np.linalg.norm(axis) #
    angle = np.arccos(normal_vec[2])
    return np.array(p.getQuaternionFromAxisAngle(axis, angle))


  @property
  def ground_orientation_world_frame(self) -> Sequence[float]:
    return np.array(
        p.multiplyTransforms([0., 0., 0.], self._robot.GetTrueBaseOrientation(),
                             [0., 0., 0.],
                             self.ground_orientation_robot_frame)[1])
