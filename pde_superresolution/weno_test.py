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

if __name__ == '__main__':
  absltest.main()
