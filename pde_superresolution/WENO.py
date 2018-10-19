import numpy as np
from .equations import Grid, RandomForcing, ConservativeBurgersEquation, staggered_first_derivative

class WENO(object):

  def __init__(self, k: int, dx: float):
    if k not in (2, 3):
      raise NotImplementedError('WENO is only implemented for k=2,3')
    self.k = k
    self.dx = dx
    self.epsilon = 1e-6

    # precompute coefficients
    self.c = {r: np.array([WENO._c(r, j, self.k) for j in range(self.k)]) for r in range(-1, self.k)}
    self.d = {r: np.array([WENO._d(r, j, self.k) for j in range(self.k)])/self.dx for r in range(-1, self.k)}

    if k == 2:
      self.gamma = np.array([2, 1]) / 3
    if k == 3:
      self.gamma = np.array([3, 6, 1]) / 10

  '''
  Reconstruction coefficients for the function value. Identifies with Table 2.1 in:
  Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws.
  Shu, Chi-Wang
  Advanced numerical approximation of nonlinear hyperbolic equations. Springer, Berlin, Heidelberg, 1998.
  https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf
  '''
  @staticmethod
  def _c(r: int, j: int, k: int):
    return np.sum([
      np.sum([
        np.prod([
          r - q + 1 for q in range(k + 1) if q != m and q != l
        ])
        for l in range(k + 1) if m != l]
      ) / np.prod([(m - l) for l in range(k + 1) if l != m])
      for m in range(j + 1, k + 1)
    ])

  '''
  Reconstruction coefficients for the function first derivative. Based on _c but homebrewed with no reference.
  '''
  @staticmethod
  def _d(r: int, j: int, k: int):
    return np.sum([
      np.sum([
        np.sum([
          np.prod([
            r - q + 1 for q in range(k + 1) if q != m and q != l and q != alpha
          ]) for alpha in range(k + 1) if alpha != m and alpha != l])
        for l in range(k + 1) if m != l]
      ) / np.prod([(m - l) for l in range(k + 1) if l != m])
      for m in range(j + 1, k + 1)
    ])

  '''
  Calculates the smoothness estimators returns an array of size (k+1, length(y)).
  The (i,r) element corresponds to smoothness estimates of cell j with offset r.
  '''

  def smoothness(self, y):
    if self.k == 2:
      return np.array([
        (np.roll(y, 1) - y) ** 2,
        (np.roll(y, -1) - y) ** 2
      ])
    elif self.k == 3:
      der2 = 13 / 12 * (y - 2 * np.roll(y, -1) + np.roll(y, -2)) ** 2

      return np.array([
        # np.roll(der2,-1) + 0.25*(5*np.roll(y,-1)-8*np.roll(y,-2)+3*np.roll(y,-3)), # for r=-1. experimental
        der2 + 0.25 * (3 * y - 4 * np.roll(y, -1) + np.roll(y, -2)) ** 2,
        np.roll(der2, 1) + 0.25 * (np.roll(y, -1) - np.roll(y, 1)) ** 2,
        np.roll(der2, 2) + 0.25 * (np.roll(y, 2) - 4 * np.roll(y, 1) + 3 * y) ** 2
      ])

  def weights(self, y):
    betas = self.smoothness(y);
    w_left, w_right = (g[:, np.newaxis] / (self.epsilon + betas) ** 2 for g in (self.gamma[::-1], self.gamma))
    w_left, w_right = (w / w.sum(axis=0) for w in (w_left, w_right))
    return w_left, w_right

  def reconstruct(self, y):
    w_left, w_right = self.weights(y)
    left, right, leftd, rightd = self.all_reconstructions(y)
    return [np.sum(q, axis=0) for q in [left * w_left, right * w_right, leftd * w_left, rightd * w_right]]

  def all_reconstructions(self, y):
    right = np.array([
      sum([
        self.c[r][j] * np.roll(y, r - j) for j in range(self.k)
      ]) for r in range(self.k)
    ])
    left = np.array([
      sum([
        self.c[r - 1][j] * np.roll(y, r - j) for j in range(self.k)
      ]) for r in range(self.k)
    ])
    rightd = np.array([
      sum([
        self.d[r][j] * np.roll(y, r - j) for j in range(self.k)
      ]) for r in range(self.k)
    ])
    leftd = np.array([
      sum([
        self.d[r - 1][j] * np.roll(y, r - j) for j in range(self.k)
      ]) for r in range(self.k)
    ])
    left = np.roll(left, -1)
    return left, right, leftd, rightd
