import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree

from ..Metrics.IMetric import *
from ..Metrics.MetricsFactory import *

__all__ = ['IMethodOfGetNeighbours']

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