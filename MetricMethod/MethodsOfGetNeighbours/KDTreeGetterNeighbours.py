from .IMethodOfGetNeighbours import *
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd

__all__ = ['KDTreeGetterNeighbours']

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