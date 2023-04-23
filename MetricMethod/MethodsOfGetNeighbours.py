import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree

from .Metrics import *

__all__ = [
    'IMethodOfGetNeighbours',
    'ExhaustiveSearchGetterNeighbours',
    'KDTreeGetterNeighbours',
    'MethodOfGetNeighboursFactory']

class IMethodOfGetNeighbours(ABC):
    
    _metric : str
    _metric_factory = MetricsFactory()

    def __init__(self, metric : str) -> None:
        self._metric = metric

    @abstractmethod
    def preprocessing(self, data : pd.DataFrame) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def get_neighbours(self, point : pd.Series, knn : int,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    

class ExhaustiveSearchGetterNeighbours(IMethodOfGetNeighbours):

    __data : pd.DataFrame

    def preprocessing(self, data: pd.DataFrame) -> None:
        self.__data = data

    def get_neighbours(self, point: pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:

        distances = self._metric_factory.get_metrics(self._metric).get_distances(self.__data, point)
        if width == -1 and knn != -1: 
            nearest_neighbor_index = np.argpartition(distances, knn, axis=None)[:knn] 
            return nearest_neighbor_index, distances
        elif width != -1 and knn == -1:
            index_elem_less_width = np.where(distances < width)[0]
            return index_elem_less_width, np.take(distances, index_elem_less_width)
        
        


class KDTreeGetterNeighbours(IMethodOfGetNeighbours):
    
    __kdtree : KDTree

    def preprocessing(self, data: pd.DataFrame) -> None:
        self.__kdtree = KDTree(data, metric=self._metric)

    def get_neighbours(self, point: pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:

        if width == -1 and knn != -1:
            distance, nearest_neighbor_index = self.__kdtree.query(point.reshape(1, -1), k=knn)
            return nearest_neighbor_index.ravel(), distance.ravel() 
        elif width != -1 and knn == -1:
            # Define a condition
            nearest_neighbor_index, distance = self.__kdtree.query_radius(point.reshape(1, -1), r=width, return_distance=True)
            return nearest_neighbor_index.ravel()[0].ravel(), distance.ravel()[0].ravel() # FIXME: I don't know why query_radius returned np.array([np.array([
        


class MethodOfGetNeighboursFactory:

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance
    
    def __del__(self):
        MethodOfGetNeighboursFactory.__instance = None

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