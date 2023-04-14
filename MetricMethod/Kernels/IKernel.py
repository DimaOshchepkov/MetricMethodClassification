import numpy as np
from abc import ABC, abstractmethod

__all__ = ['IKernel']

class IKernel(ABC):
    @abstractmethod
    def kernel_func(self, x : float) -> float:
        raise NotImplementedError()