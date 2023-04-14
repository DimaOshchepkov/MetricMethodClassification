from .IMetric import *
import numpy as np
import pandas as pd

__all__ = ['EuclideanMetric']

class EuclideanMetric(IMetric):

    def get_distances(self, data : pd.DataFrame, point : pd.Series) -> np.ndarray:
        return np.linalg.norm(data - point, axis=1)
    
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return np.linalg.norm(data - point)