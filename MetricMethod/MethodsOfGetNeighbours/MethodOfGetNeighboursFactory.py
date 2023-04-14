from .IMethodOfGetNeighbours import *

__all__ = ['MethodOfGetNeighboursFactory']

class MethodOfGetNeighboursFactory:

    __exist_methods : list = ["kdtree", "exhaustive"]

    @property
    def exist_metrics(self) -> list:
        return self.__exist_methods
    
    def method_exist(self, name : str) -> bool:
        return name in self.__exist_methods

    def get_method(self, name_method : str,
                    metric : str) -> IMethodOfGetNeighbours:
        
        if name_method == "kdtree":
            return KDTreeGetterNeighbours(metric)
        elif name_method == "exhaustive":
            return ExhaustiveSearchGetterNeighbours(metric)