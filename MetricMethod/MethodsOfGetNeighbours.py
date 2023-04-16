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
    
    __metric : str
    __metric_factory = MetricsFactory()

    def __init__(self, metric : str) -> None:
        self.__metric = metric

    @abstractmethod
    def preprocessing(self, data : pd.DataFrame) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def get_neighbours(self, data : pd.DataFrame, point : pd.Series, knn : int,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    

class ExhaustiveSearchGetterNeighbours(IMethodOfGetNeighbours):

    __data : pd.DataFrame
    __metric : str
    def __init__(self, metric : str) -> None:
        self.__metric = metric

    def preprocessing(self, data: pd.DataFrame) -> None:
        self.__data = data

    def get_neighbours(self, point: pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:

        distances = self.__metric_factory(self.__metric).get_distances(self.__data, point)
        if width == -1 and knn != -1: 
            nearest_neighbor_index = np.argpartition(distances, knn, axis=None)[:knn] 
            return nearest_neighbor_index, distances
        elif width != -1 and knn == -1:
            index_elem_less_width = np.where(distances < width)[0]
            return index_elem_less_width, np.take(distances, index_elem_less_width)
        


class KDTreeGetterNeighbours(IMethodOfGetNeighbours):
    
    __kdtree : KDTree

    def preprocessing(self, data: pd.DataFrame) -> None:
        self.__kdtree = KDTree(data, metric=self.__metric_factory(self.__metric).get_distance)

    def get_neighbours(self, point: pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:

        if width == -1 and knn != -1:
            distance, nearest_neighbor_index = self.__kdtree.query(point, k=knn,
                                                     metric=self.__metric_factory(self.__metric))
            return nearest_neighbor_index, distance
        elif width != -1 and knn == -1:
            # Define a condition
            nearest_neighbor_index = self.__kdtree.query_ball_point(point, width,
                                                     metric=self.__metric_factory(self.__metric))
            return nearest_neighbor_index, self.__kdtree[nearest_neighbor_index]
        


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