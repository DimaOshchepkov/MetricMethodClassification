from .IMetric import *
from .EuclideanMetric import *
from .CosineMetric import *
from .ManhattanMetric import *
import numpy as np
import pandas as pd

__all__ = ['MetricsFactory']

class MetricsFactory:

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