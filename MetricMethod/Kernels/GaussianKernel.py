from .IKernel import *
import numpy as np

__all__ = ['GaussianKernel']

class GaussianKernel(IKernel):

    def kernel_func(self, r: float) -> float:
        return 1/np.sqrt(2 * np.pi) * np.exp(-2 * np.power(r, 2))