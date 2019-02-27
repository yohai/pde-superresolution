# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for WENO reconstruction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np

from pde_superresolution import weno


class WENOTest(parameterized.TestCase):

  def test_calculate_omega_smooth(self):
    u = np.linspace(0, 1, num=100)
    actual = weno.calculate_omega(u)
    actual = actual[:, 2:-2]
    expected = np.stack(actual.shape[1] * [[0.1, 0.6, 0.3]], axis=1)
    np.testing.assert_allclose(actual, expected)

  def test_left_coefficients_smooth(self):
    u = np.linspace(0, 1, num=100)
    actual = weno.left_coefficients(u)
    actual = actual[:, 2:-2]  # trim edges near discontinuity (becuase of implicit perdiodic boundary condition)
    expected = np.stack(actual.shape[1] * [[2 / 60, -13 / 60, 47 / 60, 27 / 60, -3 / 60]], axis=1)
    np.testing.assert_allclose(actual, expected)

  def test_right_coefficients_smooth(self):
    u = np.linspace(0, 1, num=100)
    actual = weno.right_coefficients(u)
    actual = actual[:, 2:-3]     # trim edges near discontinuity (becuase of implicit perdiodic boundary condition)
    expected = np.stack(actual.shape[1] * [[-3/60, 27/60, 47/60, -13/60, 2/60]], axis=1)
    np.testing.assert_allclose(actual, expected)

  def test_reconstruct_left_discontinuity(self):
    u = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1])
    actual = weno.reconstruct_left(u)
    expected = [0.5, 1.5, 2.5, 3.5, 4.5, -3.5, -2.5, -1.5, -0.5]
    np.testing.assert_allclose(actual, expected, atol=0.005)

  def test_reconstruct_right_discontinuity(self):
    u = np.array([0, 1, 2, 3, 4, -4, -3, -2, -1])
    actual = weno.reconstruct_right(u)
    expected = [0.5, 1.5, 2.5, 3.5, -4.5, -3.5, -2.5, -1.5, -0.5]
    np.testing.assert_allclose(actual, expected, atol=0.005)

  @parameterized.parameters(
      dict(u=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
      dict(u=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
      dict(u=[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]),
      dict(u=[0, 0, 1, 2, 3, 0, 0, 0, 0, 0]),
      dict(u=[0, 0, 0, 1, 2, 0, 0, 0, 0, 0]),
      dict(u=2 * np.random.RandomState(0).rand(10)),
  )
  def test_reconstruction_symmetry(self, u):
    u = np.array(u, dtype=float)

    def flip(x):
      return x[::-1]

    def flip_staggered(x):
      return flip(np.roll(x, +1))

    left_direct = weno.reconstruct_left(u)
    left_flipped = flip_staggered(weno.reconstruct_right(flip(u)))
    np.testing.assert_allclose(left_direct, left_flipped, atol=1e-6)

    right_direct = weno.reconstruct_right(u)
    right_flipped = flip_staggered(weno.reconstruct_left(flip(u)))
    np.testing.assert_allclose(right_direct, right_flipped, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
