from .IKernel import *

__all__ = ['KernelFactory']

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
