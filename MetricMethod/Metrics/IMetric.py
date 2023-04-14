import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

__all__ = ['IMetric']

class IMetric(ABC):

    @abstractmethod
    def get_distances(self, data : pd.DataFrame, point : pd.Series) -> np.ndarray:
        pass

    @abstractmethod
    def get_distance(self, data : pd.Series, point : pd.Series) -> float:
        pass