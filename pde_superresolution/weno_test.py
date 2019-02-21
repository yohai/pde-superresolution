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
"""Tests for model functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np
from pde_superresolution import equations
from pde_superresolution.weno import WENO

from pde_superresolution import model  # pylint: disable=g-bad-import-order


class WenoTest(parameterized.TestCase):

  def test_symmetry(self):
    u = np.random.randn(40)
    ur = u[::-1]

    eq = equations.ConservativeBurgersEquation(len(u))
    weno = WENO(equation=eq)

    # assert that reconstruction is OK
    up, um = weno.u_plus_minus(u)
    upr, umr = weno.u_plus_minus(ur)
    assert np.allclose(up, np.roll(umr[::-1], -1))
    assert np.allclose(um, np.roll(upr[::-1], -1))

    # assert that time derivative is invariant to flipping
    # x -> -x and u -> -u
    ut = weno.calculate_time_derivative(u)
    utr = weno.calculate_time_derivative(-ur)
    assert np.allclose(ut, -utr[::-1])

  def test_curvature_calculation(self):
    u = np.random.randn(40)


def ExplicitLectureNotesWENO(v):
  '''
  This follows Procedure 2.2 as verbatim and as explicitly as possible.
   It is for test purposes and is probably very inefficient
  '''
  epsilon = 1e-6

  # Table 2.1 in the lecture notes for k=3
  _crj_matrix = {
    -1: np.array([11 / 6, -7 / 6, 1 / 3]),
    0: np.array([1 / 3, 5 / 6, -1 / 6]),
    1: np.array([-1 / 6, 5 / 6, 1 / 3]),
    2: np.array([1 / 3, -7 / 6, 11 / 6]),
  }

  def c_rj(r, j):
    return _crj_matrix[r][j]

  def cbar_rj(r, j):  # see unnumbered equation below Eq. (2.10)
    return c_rj(r - 1, j)

  ######################################################################################################
  # 1. Obtain the k reconstructed values ... in (2.51), based on the stencils (2.51) for r = 0, ..., k-1
  n = len(v)
  v_r_i_plus_half = {k: sum(c_rj(r, j) * v[ (i - r + j) % n] for j in range(k)) for r in range(k)}

  # Also obtain the k reconstructed values ...
  v_r_i_minus_half = {k: sum(cbar_rj(r, j) * v[(i - r + j) % n] for j in range(k)) for r in range(k)}

  ######################################################################################################
  # 2. Find the constants d_r such that ...
  d = np.array([3/10, 3/5, 1/10])
  # by symmetry, dbar[r] = d[k-1-r]
  dbar = d[::-1]

  ######################################################################################################
  # 3. Find the smooth indicators βr in (2.61)...
  beta_1 = (
          (13 / 12) * np.array([u[i] - 2 * u[(i + 1) % n] + u[(i + 2) % n] for i in range(n)]) ** 2 +
          (1 / 4) * np.array([3 * u[i] - 4 * u[(i + 1) % n] + u[(i + 2) % n] for i in range(n)]) ** 2
  )
  beta_2 = (
          (13 / 12) * np.array([u[(i - 1) % n] - 2 * u[(i) % n] + u[(i + 1) % n] for i in range(n)]) ** 2 +
          (1 / 4) * np.array([u[(i - 1) % n] - u[(i + 1) % n] for i in range(n)]) ** 2
  )
  beta_3 = (
          (13 / 12) * np.array([u[(i - 2) % n] - 2 * u[(i - 1) % n] + u[(i) % n] for i in range(n)]) ** 2 +
          (1 / 4) * np.array([u[i - 2] - 4 * u[(i - 1) % n] + 3 * u[(i) % n] for i in range(n)]) ** 2
  )
  beta_r = np.stack([beta_1, beta_2, beta_3])

  ######################################################################################################
  # 4. Form the weights ωr and ωr using (2.58)-(2.59) and ...
  alpha_r = d/(epsilon + beta_r)
  omega_r= alpha_r/alpha_r.sum()

  alphabar_r =  dbar/(epsilon + beta_r)
  omegabar_r = alphabar_r / alphabar_r.sum()

  ######################################################################################################
  # 5. Find the (2k − 1)-th order reconstruction ...
  v_minus_i_plus_half = sum(omega_r[r] * v_r_i_plus_half[r] for r in range(k))
  v_plus_i_minus_half = sum(omegabar_r[r] * v_r_i_minus_half[r] for r in range(k))


if __name__ == '__main__':
  absltest.main()
