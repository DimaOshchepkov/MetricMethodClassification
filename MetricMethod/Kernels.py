import numpy as np
from abc import ABC, abstractmethod

__all__ = [
    'IKernel',
    'GaussianKernel',
    'KernelFactory',
    'RectangularKernel']

class IKernel(ABC):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance
    
    def __del__(self):
        IKernel.__instance = None
        
    @abstractmethod
    def kernel_func(self, x : float) -> float:
        raise NotImplementedError()
    


class GaussianKernel(IKernel):

    def kernel_func(self, r: float) -> float:
        return 1/np.sqrt(2 * np.pi) * np.exp(-2 * np.power(r, 2))
    

class KernelFactory():
    
    __exist_kernel : list = ["rectangular", "gaussian"]

    @property
    def exist_kernel(self) -> list:
        return self.__exist_kernel
    
    def kernel_exist(self, name : str) -> bool:
        return name in self.__exist_kernel

    def get_kernel(self, name_kernel : str) -> IKernel:
        if name_kernel == "rectangular":
            return RectangularKernel()
        elif name_kernel == "gaussian":
            return GaussianKernel()
        else:
            raise ValueError("Unknown kernel")
        

class RectangularKernel(IKernel):

    def kernel_func(self, r: float) -> float:
        if np.abs(r) <= 1:
            return 0.5
        else:
            return 0