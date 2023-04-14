from .IMetric import *
import numpy as np
import pandas as pd

__all__ = ['CosineMetric']

class CosineMetric(IMetric):

    def get_distances(self, data : pd.DataFrame, point : pd.Series) -> np.ndarray:   
        return (np.ones(len(data)) - data.dot(point) / 
                (np.linalg.norm(data, axis=1) * np.linalg.norm(point)))
        
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        return (1 - data.dot(point) / 
                    (np.linalg.norm(data) * np.linalg.norm(point)))