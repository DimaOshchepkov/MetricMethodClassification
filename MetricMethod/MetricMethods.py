from .Kernels import *
from .MethodsOfGetNeighbours import *
from .Metrics import *
from abc import abstractmethod
import numpy as np
import pandas as pd

__all__ = [
    'IMetricMethod', 
    'OneNN', 
    'KNN', 
    'ParzenWindowFixedWidth', 
    'ParzenWindowVariableWidth']

class IMetricMethod():

    __method : IMethodOfGetNeighbours
    __methods_factory : MethodOfGetNeighboursFactory

    __metric : IMetric
    __metric_factory : MetricsFactory

    def __init__(self, metric : str, method : str) -> None:
        
        self.__metric_factory = MetricsFactory()
        self.__metric = self.__metric_factory.get_metrics(metric)

        self.__methods_factory = MethodOfGetNeighboursFactory()
        self.__method = self.__methods_factory.get_method(name_method=method, metric=metric)

    def fit(self, data : pd.Series) -> None:
        pass

    @abstractmethod
    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X_test : pd.DataFrame, Y_train : pd.Series) -> np.ndarray:
        raise NotImplementedError()
    

class OneNN(IMetricMethod):

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive") -> None:
        super().__init__(metric, method)


    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, knn=1)
        
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]  
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame, Y_train : pd.Series) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, 1))

        return np.array(predict) 
    


class KNN(IMetricMethod):

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive") -> None:
        super().__init__(metric, method)

    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series, countNeighbor : int) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, knn=countNeighbor)
       
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]  
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame,
                Y_train : pd.Series, count_neigbors : int = 10) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, count_neigbors))

        return np.array(predict) 
    

class ParzenWindowFixedWidth(IMetricMethod):

    __kernel : IKernel
    __kernel_factory = KernelFactory()

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "default") -> None:
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)

    def __Get_Neighbor(self, train_Y : pd.Series,
                        data_point : pd.Series, width : float) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, width=width)

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for i in nearest_index:
            nearest[train_Y.iloc[i]] += self.__kernel.kernel_func(distances[i]/width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame,
                Y_train : pd.Series, width : float = 10) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, width))

        return np.array(predict) 
    

class ParzenWindowVariableWidth(IMetricMethod):

    __kernel : IKernel
    __kernel_factory = KernelFactory()

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "default") -> None:
                 
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)

    def __Get_Neighbor(self, train_Y : pd.Series,
                        data_point : pd.Series, k : int) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, knn=k+1)
        width = distances[k]

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for i in nearest_index[:k+1]:
            nearest[train_Y.iloc[i]] += self.__kernel.kernel_func(distances[i]/width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame,
                Y_train : pd.Series, k : int = 10) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, k))

        return np.array(predict) 