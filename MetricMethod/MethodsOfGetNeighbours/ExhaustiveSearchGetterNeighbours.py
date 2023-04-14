from .IMethodOfGetNeighbours import *
import numpy as np
import pandas as pd

__all__ = ['ExhaustiveSearchGetterNeighbours']

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