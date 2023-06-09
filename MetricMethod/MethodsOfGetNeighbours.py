from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from .Metrics import *

__all__ = [
    'IMethodOfGetNeighbours',
    'ExhaustiveSearchGetterNeighbours',
    'KDTreeGetterNeighbours',
    'MethodOfGetNeighboursFactory']

class IMethodOfGetNeighbours(ABC):
    """Common interface for getting neighbours of a given point."""
    _metric : str
    _metric_factory = MetricsFactory()

    def __init__(self, metric : str) -> None:
        self._metric = metric

    @abstractmethod
    def preprocessing(self, data : pd.DataFrame) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def get_neighbours(self, point : pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:
        """
        if width == -1 and knn != -1 will then return knn nearest neighbors.
        if width != -1 and knn == -1 will then return the nearest neighbors
        within the radius width.
        Other configurations are not possible
        
        Args:
            point (pd.Series): classification point
            knn (int): count neighbours. Defaults to -1.
            width (float): width of window. Defaults to -1.


        Returns:
            tuple[np.ndarray, np.ndarray]: nearest_neighbor_index, distances.
        """
        raise NotImplementedError()
    

class ExhaustiveSearchGetterNeighbours(IMethodOfGetNeighbours):
    """Exhaustive search for neighbours of a given point."""

    @property
    def data(self):
        return self.__data
    
    __data : np.ndarray

    def preprocessing(self, data: pd.DataFrame) -> None:
        self.__data = np.array(data)

    def get_neighbours(self, point: pd.Series, knn : int = -1,
                        width : float = -1) -> tuple[np.ndarray, np.ndarray]:

        distances = self._metric_factory.get_metrics(self._metric).get_distance(self.__data, point)
        if width == -1 and knn != -1: 
            nearest_neighbor_index = np.argpartition(distances, knn, axis=None)[:knn] 
            return nearest_neighbor_index, distances
        elif width != -1 and knn == -1:
            index_elem_less_width = np.where(distances < width)[0]
            return index_elem_less_width, np.take(distances, index_elem_less_width)
        
        


class KDTreeGetterNeighbours(IMethodOfGetNeighbours):
    """Using KDtree for getting neighbours of a given point."""
    __kdtree : KDTree

    @property
    def data(self):
        return self.__kdtree

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
    """Use this class to create a method of get neighbours."""

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
        """Use this function to create a method of get neighbours.

        Args:
            name_method (str): name of the method.
            Possible values: kdtree, exhaustive.

            metric (str): metric to use.

        Returns:
            IMethodOfGetNeighbours: method of get neighbours.
        """        
        
        if name_method == "kdtree":
            return KDTreeGetterNeighbours(metric)
        elif name_method == "exhaustive":
            return ExhaustiveSearchGetterNeighbours(metric)