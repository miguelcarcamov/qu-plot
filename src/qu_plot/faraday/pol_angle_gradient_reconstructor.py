from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .faraday_reconstructor import FaradayReconstructor


@dataclass
class PolAngleGradientReconstructor(FaradayReconstructor):
    def __post_init__(self):
        super().__init__()

    @staticmethod
    def __line(p, x):
        m, c = p
        return m * x + c

    @staticmethod
    def __dtheta(a, b):
        x1 = np.abs(a - b)
        x2 = 2 * np.pi - np.abs(a - b)
        x = np.vstack((x1, x2))
        return np.amin(x, axis=0)

    def __fit_chi(self, x, y, err, p0):
        nll = lambda *args: np.sum(self.__dtheta(y, self.__line(*args)) ** 2 / err**2)

        initial = p0 + 0.01 * np.random.randn(2)
        bnds = ((None, None), (-np.pi, np.pi))

        soln = minimize(nll, initial, bounds=bnds, args=x)
        p = soln.x

        return p

    def config_fd_space(self):
        pass

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
