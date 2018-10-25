import numpy as np

_p = (1 / 6) * np.array([
	[2, -7, 11, 0, 0],
	[0, -1, 5, 2, 0],
	[0, 0, 2, 5, -1]
])
_B1 = np.array([
	[1, -2, 1, 0, 0],
	[0, 1, -2, 1, 0],
	[0, 0, 1, -2, 1]
])
_B2 = np.array([
	[1, -4, 3, 0, 0],
	[0, 1, 0, -1, 0],
	[0, 0, 3, -4, 1]
])
_d = np.array([[0.1, 0.6, 0.3]]).T

class WENO(object):
	"""
	An implementation of 4th order finite volume WENO
	"""

	def __init__(self, dx, flux=lambda x: -0.5 * x ** 2, dflux=lambda x: -x):
		"""Constructor.
				Args:
					dx: grid spacing
					flux, dflux: callables that return the physical flux f(w) and its
					derivative f'(w)
		"""
		self.flux = flux
		self.dflux = dflux
		self.eps = 1e-6
		self.dx = dx

	def flux_divergence(self, w: np.ndarray) -> np.ndarray:
		"""
		returns the WENO approximation of the divergence of the flux
		"""
		a = max(abs(self.dflux(w)))
		f = self.flux(w)
		v = 0.5 * (f + a * w)
		u = np.roll(0.5 * (f - a * w), -1)

		# right flux ######################
		vs = np.stack([np.roll(v, k) for k in reversed(range(-2, 3))])

		# polynomial reconstructions
		ps = _p @ vs

		# smoothness indicators
		Bs = (13 / 12) * (_B1 @ vs) ** 2 + 0.25 * (_B2 @ vs) ** 2

		alphas = _d / (Bs + self.eps) ** 2
		weights = alphas / alphas.sum(axis=0)
		hn = np.sum(weights * ps, axis=0)

		# left flux ######################
		us = np.stack([np.roll(u, k) for k in reversed(range(-2, 3))])
		# polynomial reconstructions
		ps = _p[::-1, ::-1] @ us

		# smoothness indicators
		Bs = (13 / 12) * (_B1 @ us) ** 2 + 0.25 * (_B2 @ us) ** 2

		alphas = _d[::-1] / (Bs + self.eps) ** 2
		weights = alphas / alphas.sum(axis=0)
		hp = np.sum(weights * ps, axis=0)

		return (hp - np.roll(hp, 1) + hn - np.roll(hn, 1)) / self.dx
