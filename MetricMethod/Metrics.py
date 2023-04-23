import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

__all__ = [
            'IMetric',
            'MetricsFactory',
            'ManhattanMetric',
            'EuclideanMetric',
            'CosineMetric']

class IMetric(ABC):

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance
    
    def __del__(self):
        IMetric.__instance = None
    

    @abstractmethod
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        """
        Args:
            data (pd.Series): vector1
            point (pd.Series): vector2

        Returns:
            float: distance between data and point
        """        
        pass


class ManhattanMetric(IMetric):
    
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return np.sum(np.abs(data - point), axis=-1)
    
class EuclideanMetric(IMetric):
    
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return np.linalg.norm(data - point, axis=-1)
    

class CosineMetric(IMetric):
       
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return (1 - data.dot(point) / 
                    (np.linalg.norm(data, axis=-1) * np.linalg.norm(point))) 
    

class MetricsFactory:

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance
    
    def __del__(self):
        MetricsFactory.__instance = None

    __exist_metrics : list = ["euclidean", "cityblock", 'cosine']

    @property
    def exist_metrics(self) -> list:
        return self.__exist_metrics
        
    def metrics_exist(self, name : str) -> bool:
        return name in self.__exist_metrics

    def get_metrics(self, name : str) -> IMetric:
        if name == "euclidean":
            return EuclideanMetric()
        elif name == "cityblock":
            return ManhattanMetric()
        elif name == 'cosine':
            return CosineMetric()