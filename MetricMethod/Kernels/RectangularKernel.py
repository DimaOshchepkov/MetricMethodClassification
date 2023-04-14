from .IKernel import *
import numpy as np

__all__ = ['RectangularKernel']

class RectangularKernel(IKernel):

    def kernel_func(self, r: float) -> float:
        if np.abs(r) <= 1:
            return 0.5
        else:
            return 0