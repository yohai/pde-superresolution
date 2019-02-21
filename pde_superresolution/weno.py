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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np

from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import polynomials  # pylint: disable=g-bad-import-order
from typing import Tuple

# constants for k=2 and k=3
_RECONSTRUCTION_COEFF = {
    2: np.array([
        [-1/2, 3/2, 0],
        [0, 1/2, 1/2],
    ]),
    3: (1 / 6) * np.array([
        [2, -7, 11, 0, 0],
        [0, -1, 5, 2, 0],
        [0, 0, 2, 5, -1]
    ]),
}
_SMOOTHNESS_COEFF = {
    2: np.array([[
        [1, -1, 0],
        [0, 1, -2],
    ]]),
    3: np.array([
        np.sqrt(13/12)*np.array([
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
        ]),
        0.5*np.array([
            [1, -4, 3, 0, 0],
            [0, 1, 0, -1, 0],
            [0, 0, 3, -4, 1],
        ]),
    ])
}
_WEIGHTS_D = {
    2: np.array([[1/3, 2/3]]).T,
    3: np.array([[0.1, 0.6, 0.3]]).T,
}

UX_4_POINT = polynomials.coefficients(
    np.array([1.5, 0.5, -0.5, -1.5]),
    polynomials.Method.FINITE_VOLUMES,
    derivative_order=1)


class FluxMethod(enum.Enum):
  """Relationship between successive grids."""
  GODUNOV = 1
  LAX_FRIEDRICHS = 2


class WENO(object):
  """Finite volume WENO implementation.

  An implementation of 4th order finite volume WENO, following Chi-Wang Shu,
  "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes
  for Hyperbolic Conservation Laws", NASA/CR-97-206253, Nov 1997,
  available at https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf

  In their notation, this implementation uses k=3.
  """

  def __init__(
      self,
      equation: equations.ConservativeBurgersEquation,
      flux_method: FluxMethod = FluxMethod.GODUNOV,
      k: int = 3):
    """Constructor.

    Args:
      equation: conservative burgers equation to integrate.
      flux_method: monotone flux method.
      k: reconstruction order.
    """
    if not isinstance(equation, equations.ConservativeBurgersEquation):
      raise TypeError('invalid equation: {}'.format(equation))

    self.equation = equation
    self.flux_method = flux_method
    self.eps = 1e-6
    self.dx = equation.grid.solution_dx
    self.k = k

  def calculate_time_derivative(self, u: np.ndarray) -> np.ndarray:
    """Returns the WENO approximation of the divergence of the flux."""
    # reconstrution at +1/2 cells
    u_plus, u_minus = self.u_plus_minus(u)

    if self.k == 2:
      u_x = (np.roll(u, -1) - u) / self.dx
    elif self.k == 3:
      u_x = (np.roll(u, -2) * UX_4_POINT[0]
             + np.roll(u, -1) * UX_4_POINT[1]
             + u * UX_4_POINT[2]
             + np.roll(u, 1) * UX_4_POINT[3]) / self.dx

    if self.flux_method is FluxMethod.LAX_FRIEDRICHS:
      # flux at +1/2 cells
      f_minus = self.equation.flux(u_minus, u_x)
      f_plus = self.equation.flux(u_plus, u_x)

      # NOTE(shoyer): in principle, I think we could replace this by a local
      # maximum, but I doubt that would make a difference for Burgers' equation
      alpha = np.max(abs(self.equation.flux_derivative(u)))
      flux = 0.5 * (f_minus + f_plus - alpha * (u_plus - u_minus))

    elif self.flux_method is FluxMethod.GODUNOV:
      w = np.linspace(0, 1, num=1001).reshape(-1, 1)
      u_range = w * u_minus + (1 - w) * u_plus
      f_range = self.equation.flux(u_range, u_x)
      f_min = f_range.min(axis=0)
      f_max = f_range.max(axis=0)
      flux = np.where(u_minus <= u_plus, f_min, f_max)
    else:
      raise ValueError('invalid flux method')

    # difference of flux at +1/2 and -1/2 cells
    time_deriv = (flux - np.roll(flux, 1)) / self.dx
    return time_deriv

  def u_plus_minus(self, u) -> Tuple[np.ndarray, np.ndarray]:
      u_minus = self.reconstruction(np.roll(u, -1))
      u_plus = self.reconstruction(u[::-1])[::-1]
      return u_plus, u_minus

  def reconstruction(self, u: np.ndarray) -> np.ndarray:
    """Reconstructs the flux from the point values f.

    Args:
      u: input values for the scalar field.
      p: polynomial coefficients.
      s: coefficients to use in calculating smoothness.
      d: optimal convex combination coefficients for smooth functions (Eq.
        (2.54) in the lecture notes).

    Returns:
      Reconstructed flux.
    """
    p = _RECONSTRUCTION_COEFF[self.k]
    s = _SMOOTHNESS_COEFF[self.k]
    d = _WEIGHTS_D[self.k]
    # defined at [+2.5, +1.5, +0.5, -0.5, -1.5] cells (for k=3)
    ws = np.stack([np.roll(u, k) for k in range(-(self.k - 1), self.k)])
    # polynomial reconstruction
    ps = np.matmul(p, ws)
    # smoothness indicators
    beta_r = np.sum(np.einsum('ijk,kl->ijl', s, ws) ** 2, axis=0)
    alphas = d / (beta_r + self.eps) ** 2
    weights = alphas / alphas.sum(axis=0)
    return np.sum(weights * ps, axis=0)
