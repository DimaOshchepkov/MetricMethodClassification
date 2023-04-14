from .IMetric import *
import numpy as np
import pandas as pd

__all__ = ['ManhattanMetric']

class ManhattanMetric(IMetric):

    def get_distances(self, data : pd.DataFrame, point : pd.Series) -> np.ndarray:
        return np.sum(np.abs(data - point), axis=1)
    
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return np.sum(np.abs(data - point))