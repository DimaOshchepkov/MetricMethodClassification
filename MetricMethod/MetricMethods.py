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
    """Is a generalization for all metric methods"""
    _method : IMethodOfGetNeighbours
    _methods_factory : MethodOfGetNeighboursFactory

    _metric : IMetric
    _metric_factory : MetricsFactory

    _y_train : pd.Series

    def __init__(self, metric : str, method : str) -> None:
        """Common initializer for all metric methods

        Args:
            metric (str): kind of metric. 
            Possible values: "euclidean", "manhattan", "cosine".

            method (str): nearest neighbor method. 
            Possible values: "exhaustive", "kdtree".
            
        """
        self._metric_factory = MetricsFactory()
        self._metric = self._metric_factory.get_metrics(metric)

        self._methods_factory = MethodOfGetNeighboursFactory()
        self._method = self._methods_factory.get_method(name_method=method, metric=metric)

    @abstractmethod
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        """Data preprocessing

        Args:
            data (pd.Series): initial data
            y_train (pd.Series): training data class labels
        """
        raise NotImplementedError()

    @abstractmethod
    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:
        """
        Args:
            train_Y (pd.Series): Training sample class labels 
            data_point (pd.Series): The object for which the nearest neighbor is found

        Returns:
            any: Returns nearest neighbor
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        """
        Args:
            X_test (pd.DataFrame): test set
            Y_train (pd.Series): training data class labels

        Returns:
            np.ndarray: Array of predicted class labels
        """
        raise NotImplementedError()
    

class OneNN(IMetricMethod):

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive") -> None:
        super().__init__(metric, method)


    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:

        nearest_index, distances = self._method.get_neighbours(point=data_point, knn=1)
        
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]  
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:

        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(self._y_train, row, 1))

        return np.array(predict) 
    


class KNN(IMetricMethod):

    __countNeighbor : int

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 countNeigbor : int = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            Possible values: "euclidean", "manhattan", "cosine".

            method (str): nearest neighbor method. Defaults to "exhaustive". 
            Possible values: "exhaustive", "kdtree".
            
            countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """
        super().__init__(metric, method)
        self.__countNeighbor = countNeigbor    

    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:      

        nearest_index, distances = self._method.get_neighbours(point=data_point, knn=self.__countNeighbor)
       
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]  
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(self._y_train, row))

        return np.array(predict) 
    

class ParzenWindowFixedWidth(IMetricMethod):

    __kernel : IKernel
    __kernel_factory = KernelFactory()

    __width : float

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "rectangular", width : float = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            Possible values: "euclidean", "manhattan", "cosine".

            method (str): nearest neighbor method. Defaults to "exhaustive". 
            Possible values: "exhaustive", "kdtree".

            kernel (str): kind of kernel. Defaults to "rectangular".

            width (float): window width. Defaults to 10.
        """
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
        self.__width = width

    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:       

        nearest_index, distances = self._method.get_neighbours(point=data_point, width=self.__width)

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index, distances):
            nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/self.__width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(self._y_train, row))

        return np.array(predict) 
    

class ParzenWindowVariableWidth(IMetricMethod):

    __kernel : IKernel
    __kernel_factory = KernelFactory()

    __countNeighbor : int

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "rectangular", countNeighbor : int = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            Possible values: "euclidean", "manhattan", "cosine".

            method (str): nearest neighbor method. Defaults to "exhaustive". 
            Possible values: "exhaustive", "kdtree".

            kernel (str): kind of kernel. Defaults to "rectangular". 
            Possible values: "rectangular", "gaussian".

            countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """        
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
        self.__countNeighbor = countNeighbor

    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:

        nearest_index, distances = self._method.get_neighbours(point=data_point, knn=self.__countNeighbor + 1)
        width = distances[self.__countNeighbor]

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index[:self.__countNeighbor + 1], distances[:self.__countNeighbor + 1]):
            nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(self._y_train, row))

        return np.array(predict) 
    
    class PotentialFunction(IMetricMethod):

        __kernel : IKernel
        __kernel_factory = KernelFactory()

        __width : int
        __eps : float

        __potentials : np.ndarray
        

        def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "rectangular", width : float = 0.3, eps : float = 0.05) -> None:
            """
            Args:
                metric (str): kind of metric. Defaults to "euclidean". 
                Possible values: "euclidean", "manhattan", "cosine".
                
                method (str): nearest neighbor method. Defaults to "exhaustive".
                Possible values: "exhaustive", "kdtree".
                
                kernel (str): kind of kernel. Defaults to "rectangular".
                Possible values: "rectangular", "gaussian".
                
                countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """        
            super().__init__(metric, method)
            self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
            self.__width = width
            self.__eps = eps

        def fit(self, data : pd.Series, y_train : pd.Series) -> None:
            self._method.preprocessing(data)
            self._y_train = y_train
            self.__potentials = np.zeros(data.shape[0])

            # Distance matrix from each vector to each
            self._metric(data[:, np.newaxis, :], data[np.newaxis, :, :]) #TODO: this don't work to cosine metric
            

        def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:       

            nearest_index, distances = self._method.get_neighbours(point=data_point, width=self.__width)

            nearest = {cl : 0 for cl in np.unique(train_Y)}
            for ind, dist in zip(nearest_index, distances):
                nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/self.__width)

            return max(nearest, key=nearest.get)
        
        def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
            predict = []
            for row in np.array(X_test):
                predict.append(self.__Get_Neighbor(self._y_train, row))

            return np.array(predict)